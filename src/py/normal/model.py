"""Encoder-Decoder (seq2seq) transformer, Llama + GPT tarzi seceneklerle.

Mimari (T5/BART tarzi ama Llama bilesenleriyle):
  encoder : cift yonlu self-attention (nedensel mask YOK) + FFN, pre-norm.
  decoder : nedensel self-attention + cross-attention (encoder ciktisina) + FFN.

Llama tarafi: RMSNorm, RoPE, GQA, SwiGLU, pre-norm, bias yok.
GPT tarafi  : ogrenilmis pozisyon, LayerNorm, GELU FFN, dropout -- hepsi
              `norm_type`/`activation`/`use_rope`/`bias` ile secilebilir.
Modern      : QK-Norm (Llama 3), RoPE scaling, z-loss, KV cache (self + cross),
              gradient checkpointing, typical/mirostat ornekleme.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GenerationConfig, TrainConfig


def default_swiglu_hidden(d_model: int, multiple_of: int = 64, ffn_dim_multiplier: float = 1.0) -> int:
    """Llama kurali: ~8/3 * d_model, `multiple_of` katina yuvarlanmis."""
    hidden = int(2 * (4 * d_model) / 3)
    hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def make_norm(dim: int, cfg: TrainConfig) -> nn.Module:
    """norm_type ile RMSNorm (Llama) veya LayerNorm (GPT) secer."""
    if cfg.norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=1e-5)
    return RMSNorm(dim, eps=1e-5)


class RotaryEmbedding(nn.Module):
    """RoPE; pozisyon ofseti (KV cache) ve dogrusal olceklemeyi destekler."""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0, scaling_factor: float = 1.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE icin head_dim cift olmali")
        self.dim = dim
        self.base = base
        self.scaling_factor = max(1e-6, float(scaling_factor))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build(max_seq_len)

    def _build(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, dtype=torch.float32) / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, pos_offset: int = 0):
        seq_len = x.size(-2)
        if pos_offset + seq_len > self.max_seq_len:
            self._build(pos_offset + seq_len)
        cos = self.cos_cached[pos_offset:pos_offset + seq_len].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[pos_offset:pos_offset + seq_len].to(device=x.device, dtype=x.dtype)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class Attention(nn.Module):
    """Cok-kafali dikkat.

    * `causal=True`  -> nedensel self-attention (decoder self).
    * `cross=True`   -> q decoder'dan, k/v encoder hafizasindan (cross-attention).
    * `cache`        -> (k, v) tuple'i; generation'da kullanilir.
    * `pad_mask`     -> bool [b, s], True = pad (maskelenir).
    """

    def __init__(self, cfg: TrainConfig, causal: bool = True, cross: bool = False):
        super().__init__()
        if cfg.d_model % cfg.n_head != 0:
            raise ValueError("d_model, n_head'in kati olmali")
        self.causal = causal
        self.cross = cross
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head if cfg.n_kv_head else cfg.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head, n_kv_head'in kati olmali (GQA)")
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.q_proj = nn.Linear(cfg.d_model, self.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, context=None, cache=None, rope=None, pos_offset: int = 0, pad_mask=None):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        if self.cross:
            if cache is not None:
                k, v = cache
                new_cache = cache
            else:
                s = context.size(1)
                k = self.k_proj(context).view(b, s, self.n_kv_head, self.head_dim).transpose(1, 2)
                v = self.v_proj(context).view(b, s, self.n_kv_head, self.head_dim).transpose(1, 2)
                if self.qk_norm:
                    k = self.k_norm(k)
                new_cache = (k, v)
            if self.qk_norm:
                q = self.q_norm(q)
        else:
            k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            if rope is not None:
                cos, sin = rope(x, pos_offset=pos_offset)
                q, k = apply_rotary(q, k, cos, sin)
            if cache is not None:
                past_k, past_v = cache
                if past_k is not None and past_k.size(2) > 0:
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        key_len = k.size(2)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        if cache is not None:
            offset = key_len - t
            if self.cross and pad_mask is not None:
                # Cross-attention cache'li dahi olsa encoder dolgusunu maskele.
                allowed = (~pad_mask[:, None, None, :]).expand(-1, 1, t, key_len)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=dropout_p, is_causal=False)
            elif self.causal and t > 1 and offset > 0:
                mask = torch.ones(t, key_len, dtype=torch.bool, device=q.device).tril(diagonal=offset)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=False)
            else:
                # cross-attention: maskesiz (tum kaynagi gorur). self-attention: t==1
                # veya offset==0 ise nedensel; degilse zaten tum anahtarlar gecmis.
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=dropout_p, is_causal=(self.causal and t > 1 and offset == 0)
                )
        else:
            if self.cross:
                allowed = torch.ones(t, key_len, dtype=torch.bool, device=q.device)
            else:
                allowed = torch.ones(t, key_len, dtype=torch.bool, device=q.device).tril()
            if pad_mask is not None:
                allowed = allowed[None, None] & (~pad_mask[:, None, None, :])
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=dropout_p, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y)), new_cache


class SwiGLU(nn.Module):
    """Llama FFN: x -> down( silu(gate(x)) * up(x) )."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        hidden = cfg.ff_dim if cfg.ff_dim > 0 else default_swiglu_hidden(cfg.d_model)
        self.hidden = hidden
        self.gate = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.down = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class MLP(nn.Module):
    """GPT FFN: x -> proj( gelu(fc(x)) )."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        hidden = cfg.ff_dim if cfg.ff_dim > 0 else 4 * cfg.d_model
        self.hidden = hidden
        self.fc = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


def make_ffn(cfg: TrainConfig) -> nn.Module:
    """activation ile SwiGLU (Llama) veya GELU MLP (GPT) secer."""
    if cfg.activation == "gelu":
        return MLP(cfg)
    return SwiGLU(cfg)


class EncoderBlock(nn.Module):
    """Cift yonlu self-attention + FFN."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.ln_1 = make_norm(cfg.d_model, cfg)
        self.attn = Attention(cfg, causal=False)
        self.ln_2 = make_norm(cfg.d_model, cfg)
        self.mlp = make_ffn(cfg)
        self.grad_checkpointing = cfg.grad_checkpointing

    def _fwd(self, x, rope, pad_mask):
        x = x + self.attn(self.ln_1(x), rope=rope, pad_mask=pad_mask)[0]
        return x + self.mlp(self.ln_2(x))

    def forward(self, x, rope=None, pad_mask=None):
        if self.grad_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(lambda inp: self._fwd(inp, rope, pad_mask), x, use_reentrant=False)
        return self._fwd(x, rope, pad_mask)


class DecoderBlock(nn.Module):
    """Nedensel self-attention + cross-attention + FFN."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.ln_1 = make_norm(cfg.d_model, cfg)
        self.self_attn = Attention(cfg, causal=True)
        self.ln_2 = make_norm(cfg.d_model, cfg)
        self.cross_attn = Attention(cfg, causal=False, cross=True)
        self.ln_3 = make_norm(cfg.d_model, cfg)
        self.mlp = make_ffn(cfg)
        self.grad_checkpointing = cfg.grad_checkpointing

    def _fwd(self, x, memory, self_cache, cross_cache, rope, pos_offset, tgt_pad, mem_pad):
        a, sc = self.self_attn(self.ln_1(x), cache=self_cache, rope=rope, pos_offset=pos_offset, pad_mask=tgt_pad)
        x = x + a
        c, cc = self.cross_attn(self.ln_2(x), context=memory, cache=cross_cache, pad_mask=mem_pad)
        x = x + c
        x = x + self.mlp(self.ln_3(x))
        return x, sc, cc

    def forward(self, x, memory, self_cache=None, cross_cache=None, rope=None, pos_offset=0, tgt_pad=None, mem_pad=None):
        if self.grad_checkpointing and self.training and self_cache is None:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(
                lambda inp: self._fwd(inp, memory, None, None, rope, pos_offset, tgt_pad, mem_pad)[0],
                x, use_reentrant=False,
            )
            return out, None, None
        return self._fwd(x, memory, self_cache, cross_cache, rope, pos_offset, tgt_pad, mem_pad)


class EncoderDecoderLM(nn.Module):
    """Seq2seq transformer: kaynak metni kodlar, hedefi token-token uretir."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        head_dim = cfg.d_model // cfg.n_head
        self.rope = None
        self.pos_emb = None
        if cfg.use_rope and head_dim % 2 == 0:
            scaling = cfg.rope_scaling_factor if cfg.rope_scaling == "linear" else 1.0
            self.rope = RotaryEmbedding(head_dim, max_seq_len=cfg.block_size, base=cfg.rope_theta, scaling_factor=scaling)
        else:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)

        self.drop = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.enc_norm = make_norm(cfg.d_model, cfg)
        self.dec_norm = make_norm(cfg.d_model, cfg)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        # Residual olcek: her residual dal icin 1/sqrt(2*n) (GPT kurali). Encoder
        # blogunda 2 dal (self + FFN), decoder blogunda 3 dal (self + cross + FFN).
        enc_std = 0.02 / (2 * cfg.n_layers) ** 0.5
        dec_std = 0.02 / (3 * cfg.n_layers) ** 0.5
        for name, p in self.named_parameters():
            if p.ndim != 2:
                continue
            if name.startswith("encoder.") and name.endswith(("attn.proj.weight", "mlp.down.weight", "mlp.proj.weight")):
                nn.init.normal_(p, mean=0.0, std=enc_std)
            elif name.startswith("decoder.") and name.endswith(
                ("self_attn.proj.weight", "cross_attn.proj.weight", "mlp.down.weight", "mlp.proj.weight")
            ):
                nn.init.normal_(p, mean=0.0, std=dec_std)

    @property
    def uses_rope(self) -> bool:
        return self.rope is not None

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _pos(self, h, pos_offset):
        if self.pos_emb is not None:
            t = h.size(1)
            pos = torch.arange(pos_offset, pos_offset + t, device=h.device).unsqueeze(0).expand(h.size(0), -1)
            h = h + self.pos_emb(pos)
        return h

    def encode(self, src, src_pad=None):
        h = self._pos(self.token_emb(src), 0)
        h = self.drop(h)
        for block in self.encoder:
            h = block(h, rope=self.rope, pad_mask=src_pad)
        return self.enc_norm(h)

    def forward(
        self,
        src,
        tgt,
        targets=None,
        src_pad=None,
        tgt_pad=None,
        caches=None,
        return_cache: bool = False,
        pos_offset: int = 0,
    ):
        if src.dim() != 2 or tgt.dim() != 2:
            raise ValueError("src ve tgt [batch, seq] olmali")
        if pos_offset + tgt.size(1) > self.cfg.block_size:
            raise ValueError(f"decoder uzunlugu {pos_offset + tgt.size(1)} block_size {self.cfg.block_size}'i asiyor")

        memory = self.encode(src, src_pad)
        h = self._pos(self.token_emb(tgt), pos_offset)
        h = self.drop(h)

        new_caches = [] if return_cache else None
        for i, block in enumerate(self.decoder):
            self_cache = cross_cache = None
            if caches is not None:
                self_cache, cross_cache = caches[i]
            h, sc, cc = block(h, memory, self_cache, cross_cache, self.rope, pos_offset, tgt_pad, src_pad)
            if return_cache:
                new_caches.append((sc, cc))

        h = self.dec_norm(h)
        logits = self.head(h)

        if targets is None:
            return (logits, new_caches) if return_cache else logits

        flat_logits = logits.reshape(-1, self.cfg.vocab_size)
        flat_targets = targets.reshape(-1)
        valid = flat_targets != -100
        raw = F.cross_entropy(
            flat_logits, flat_targets, reduction="none",
            ignore_index=-100, label_smoothing=self.cfg.label_smoothing,
        )
        loss = raw.sum() / valid.sum().clamp(min=1)

        if self.cfg.z_loss > 0.0:
            log_z = torch.logsumexp(flat_logits.float(), dim=-1)
            if bool(valid.any()):
                z = (log_z[valid] ** 2).mean()
            else:
                z = log_z.new_zeros(())
            loss = loss + self.cfg.z_loss * z.to(loss.dtype)
        return logits, loss

    # ------------------------------------------------------------------
    # Ornekleme
    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _apply_sampler(next_logits, top_k, top_p, min_p, sampler, typical_p, mu):
        if sampler == "mirostat":
            probs = torch.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            surprise = -torch.log2(sorted_probs.clamp_min(1e-12))
            keep_sorted = surprise <= mu
            keep_sorted[..., 0] = True
            keep = torch.zeros_like(probs, dtype=torch.bool)
            keep.scatter_(-1, sorted_idx, keep_sorted)
            probs = probs.masked_fill(~keep, 0.0)
            return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        if top_k is not None and top_k > 0:
            topv, topi = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
            mask = torch.full_like(next_logits, float("-inf"))
            mask.scatter_(1, topi, topv)
            next_logits = mask

        probs = torch.softmax(next_logits, dim=-1)

        if sampler == "typical" and typical_p and 0.0 < typical_p < 1.0:
            logp = torch.log(probs + 1e-10)
            entropy = -(probs * logp).sum(dim=-1, keepdim=True)
            shifted = torch.abs(-logp - entropy)
            sorted_shifted, idx = torch.sort(shifted, dim=-1)
            sorted_probs = probs.gather(-1, idx)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cond = cumulative >= typical_p
            has = cond.any(dim=-1)
            first = torch.argmax(cond.float(), dim=-1)
            k = torch.where(has, first, torch.full_like(first, probs.size(-1) - 1))
            rank = torch.arange(probs.size(-1), device=probs.device).unsqueeze(0)
            keep_sorted = rank <= k.unsqueeze(-1)
            keep = torch.zeros_like(probs, dtype=torch.bool)
            keep.scatter_(-1, idx, keep_sorted)
            probs = probs.masked_fill(~keep, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        elif top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Esigi asan token'i DAHIL et: kumulatif > top_p olanlari cikar, sonra
            # bir kaydir (ilk esigi asan token kalir). Ilk token her zaman kalir.
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            keep = torch.zeros_like(probs, dtype=torch.bool)
            keep.scatter_(1, sorted_indices, ~remove)
            probs = probs.masked_fill(~keep, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        if min_p is not None and min_p > 0.0:
            min_p_value = min_p * probs.max(dim=-1, keepdim=True).values
            probs = torch.where(probs >= min_p_value, probs, torch.zeros_like(probs))
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        return probs

    def generate(
        self,
        src: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.9,
        min_p: float = 0.02,
        repetition_penalty: float = 1.2,
        device: torch.device | None = None,
        bos_id: int = 1,
        eos_id: int = 2,
        forbidden_ids=None,
        stop_ids=None,
        use_cache: bool = True,
        sampler: str = "top_p",
        typical_p: float = 0.95,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
    ):
        if src.dim() == 1:
            src = src.unsqueeze(0)

        device = device or next(self.parameters()).device
        self.eval()
        src = src.to(device)

        # Decoder toplam uzunlugu block_size'i asmasin (pos_emb tasmasi / egitimde
        # gorulmemis pozisyonlari onle). BOS + max_new_tokens <= block_size.
        max_new_tokens = max(0, min(int(max_new_tokens), self.cfg.block_size - 1))

        if forbidden_ids is None:
            forbidden_ids = {0, 1, 3}
        forbidden_ids = {int(i) for i in forbidden_ids}
        stop_set = set()
        if eos_id is not None:
            stop_set.add(int(eos_id))
        if stop_ids:
            stop_set.update(int(i) for i in stop_ids)
        forbidden_ids -= stop_set

        src_pad = src == 0
        if not bool(src_pad.any()):
            src_pad = None

        with torch.no_grad():
            memory = self.encode(src, src_pad)

            generated = [int(bos_id)]
            produced = 0
            mu = 2.0 * mirostat_tau
            caches = None
            for _ in range(max_new_tokens):
                if use_cache:
                    cur = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
                    pos = len(generated) - 1
                else:
                    cur = torch.tensor([generated], device=device, dtype=torch.long)
                    pos = 0
                h = self._pos(self.token_emb(cur), pos)
                h = self.drop(h)
                new_caches = [] if use_cache else None
                for i, block in enumerate(self.decoder):
                    self_cache = cross_cache = None
                    if use_cache and caches is not None:
                        self_cache, cross_cache = caches[i]
                    h, sc, cc = block(h, memory, self_cache, cross_cache, self.rope, pos, None, src_pad)
                    if use_cache:
                        new_caches.append((sc, cc))
                if use_cache:
                    caches = new_caches
                logits = self.head(self.dec_norm(h))[:, -1, :]

                next_logits = logits / max(temperature, 1e-8)

                if repetition_penalty is not None and repetition_penalty > 1.0 and len(generated) > 1:
                    seen = torch.tensor(generated[1:], device=device, dtype=torch.long)
                    uniq = torch.unique(seen)
                    vals = next_logits[0, uniq]
                    vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
                    next_logits[0, uniq] = vals

                full_probs = None
                if sampler == "mirostat":
                    full_probs = torch.softmax(next_logits, dim=-1)

                probs = self._apply_sampler(next_logits, top_k, top_p, min_p, sampler, typical_p, mu)

                for fid in forbidden_ids:
                    if 0 <= fid < probs.size(-1):
                        probs[..., fid] = 0.0
                if produced == 0:
                    for sid in stop_set:
                        if 0 <= sid < probs.size(-1):
                            probs[..., sid] = 0.0

                total = probs.sum(dim=-1, keepdim=True)
                if float(total.item()) <= 0.0:
                    probs = torch.ones_like(probs) / probs.size(-1)
                else:
                    probs = probs / total

                next_token = torch.multinomial(probs, num_samples=1)
                token_id = int(next_token[0].item())

                if sampler == "mirostat" and full_probs is not None:
                    p_t = float(full_probs[0, token_id].item())
                    if p_t > 0.0:
                        surprise = -math.log2(p_t)
                        mu -= mirostat_eta * (surprise - mirostat_tau)

                generated.append(token_id)
                if token_id in stop_set:
                    break
                produced += 1

        return torch.tensor([generated], device=device, dtype=torch.long)

    def generate_text(self, tokenizer, prompt: str, generation: GenerationConfig | None = None, device=None):
        from .generate import generate_text

        return generate_text(self, tokenizer, prompt, generation=generation, device=device)
