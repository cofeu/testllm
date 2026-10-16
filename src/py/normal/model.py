"""Llama tarzi decoder-only model: RMSNorm, RoPE, GQA, SwiGLU, KV cache.

Ek yetenekler:
  * RoPE (rotary positional embedding) -- uzun baglam genellemesi
  * GQA (grouped-query attention) + KV cache -- hizli uretim
  * gradient checkpointing -- dusuk bellek
  * gelismis ornekleme: top_p, typical, mirostat v2
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GenerationConfig, TrainConfig


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


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding; pozisyon ofseti ile KV cache uyumlu."""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE icin head_dim cift olmali")
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build(max_seq_len)

    def _build(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, dtype=torch.float32)
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


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        if cfg.d_model % cfg.n_head != 0:
            raise ValueError("d_model, n_head'in kati olmali")
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

    def forward(self, x, cache=None, rope=None, pos_offset: int = 0):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

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

        is_causal = t > 1 and k.size(2) == t
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y)), new_cache


class SwiGLU(nn.Module):
    """x -> down( silu(gate(x)) * up(x) ). GELU FFN yerine (Llama tarzi)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        hidden = cfg.ff_dim
        self.gate = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.down = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.grad_checkpointing = cfg.grad_checkpointing

    def _attn_mlp(self, x, cache, rope, pos_offset):
        attn_out, new_cache = self.attn(self.ln_1(x), cache=cache, rope=rope, pos_offset=pos_offset)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache

    def forward(self, x, cache=None, rope=None, pos_offset: int = 0):
        if self.grad_checkpointing and self.training and cache is None:
            from torch.utils.checkpoint import checkpoint

            def fn(inp):
                out, _ = self._attn_mlp(inp, None, rope, pos_offset)
                return out

            return checkpoint(fn, x, use_reentrant=False), None
        return self._attn_mlp(x, cache, rope, pos_offset)


class DecoderOnlyLM(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        head_dim = cfg.d_model // cfg.n_head
        self.rope = None
        self.pos_emb = None
        if cfg.use_rope and head_dim % 2 == 0:
            self.rope = RotaryEmbedding(head_dim, max_seq_len=cfg.block_size, base=cfg.rope_theta)
        else:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") and p.ndim == 2:
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        caches=None,
        return_cache: bool = False,
        pos_offset: int = 0,
    ):
        if x.dim() != 2:
            raise ValueError(f"x must be [batch, seq], got {tuple(x.shape)}")
        b, t = x.shape
        if pos_offset + t > self.cfg.block_size:
            raise ValueError(f"sequence length {pos_offset + t} exceeds block_size {self.cfg.block_size}")

        h = self.token_emb(x)
        if self.pos_emb is not None:
            pos = torch.arange(pos_offset, pos_offset + t, device=x.device).unsqueeze(0).expand(b, -1)
            h = h + self.pos_emb(pos)
        h = self.drop(h)

        new_caches = [] if return_cache else None
        for i, block in enumerate(self.blocks):
            block_cache = None if caches is None else caches[i]
            h, updated = block(h, cache=block_cache, rope=self.rope, pos_offset=pos_offset)
            if return_cache:
                new_caches.append(updated)

        h = self.norm(h)
        logits = self.head(h)

        if targets is None:
            return (logits, new_caches) if return_cache else logits

        flat_logits = logits.reshape(-1, self.cfg.vocab_size)
        flat_targets = targets.reshape(-1)
        if loss_mask is not None:
            flat_mask = loss_mask.reshape(-1).float()
            raw = F.cross_entropy(
                flat_logits, flat_targets, reduction="none",
                ignore_index=-100, label_smoothing=self.cfg.label_smoothing,
            )
            loss = (raw * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
        else:
            loss = F.cross_entropy(
                flat_logits, flat_targets, reduction="mean",
                ignore_index=-100, label_smoothing=self.cfg.label_smoothing,
            )
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
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cond = cumulative >= (1.0 - sorted_probs)
            has = cond.any(dim=-1)
            first = torch.argmax(cond.float(), dim=-1)
            k = torch.where(has, first, torch.full_like(first, probs.size(-1) - 1))
            rank = torch.arange(probs.size(-1), device=probs.device).unsqueeze(0)
            keep_sorted = rank <= k.unsqueeze(-1)
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
            mask = cumulative <= top_p
            if mask.any(dim=-1, keepdim=True).item() == 0:
                mask[..., 0] = True
            keep = torch.zeros_like(probs, dtype=torch.bool)
            keep.scatter_(1, sorted_indices, mask)
            probs = probs.masked_fill(~keep, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        if min_p is not None and min_p > 0.0:
            min_p_value = min_p * probs.max(dim=-1, keepdim=True).values
            probs = torch.where(probs >= min_p_value, probs, torch.zeros_like(probs))
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return probs

    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.9,
        min_p: float = 0.02,
        repetition_penalty: float = 1.2,
        device: torch.device | None = None,
        eos_id: int = 2,
        forbidden_ids=None,
        stop_ids=None,
        use_cache: bool = True,
        sampler: str = "top_p",
        typical_p: float = 0.95,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
    ):
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        device = device or next(self.parameters()).device
        self.eval()
        generated = prompt.to(device).clone()

        if forbidden_ids is None:
            forbidden_ids = {0, 1, 3}
        forbidden_ids = {int(i) for i in forbidden_ids}
        stop_set = set()
        if eos_id is not None:
            stop_set.add(int(eos_id))
        if stop_ids:
            stop_set.update(int(i) for i in stop_ids)
        forbidden_ids -= stop_set

        mu = 2.0 * mirostat_tau
        caches = None
        cache_len = 0
        produced = 0
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if not use_cache:
                    window = generated[:, -self.cfg.block_size:]
                    logits = self(window)[:, -1, :]
                elif caches is None or cache_len >= self.cfg.block_size:
                    window = generated[:, -self.cfg.block_size:]
                    out, caches = self(window, return_cache=True)
                    cache_len = window.size(1)
                    logits = out[:, -1, :]
                else:
                    last = generated[:, -1:]
                    out, caches = self(last, caches=caches, return_cache=True, pos_offset=cache_len)
                    cache_len += 1
                    logits = out[:, -1, :]

                next_logits = logits / max(temperature, 1e-8)

                if repetition_penalty is not None and repetition_penalty > 1.0:
                    for prev in torch.unique(generated).tolist():
                        prev = int(prev)
                        if 0 <= prev < next_logits.size(-1):
                            value = next_logits[0, prev]
                            if value > 0:
                                next_logits[0, prev] = value / repetition_penalty
                            else:
                                next_logits[0, prev] = value * repetition_penalty

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

                generated = torch.cat([generated, next_token], dim=1)

                if token_id in stop_set:
                    break
                produced += 1
        return generated

    def generate_text(self, tokenizer, prompt: str, generation: GenerationConfig | None = None, device=None):
        from .generate import generate_text

        return generate_text(self, tokenizer, prompt, generation=generation, device=device)
