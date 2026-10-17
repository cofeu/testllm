import unittest

import torch

from src.py.normal import (
    EncoderDecoderLM,
    TrainConfig,
    default_swiglu_hidden,
    make_ffn,
    make_norm,
)
from src.py.normal.model import MLP, RMSNorm, SwiGLU


def _tiny(**kw):
    base = dict(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_head=4,
        n_kv_head=2,
        ff_dim=default_swiglu_hidden(32),
        block_size=32,
        dropout=0.0,
    )
    base.update(kw)
    return TrainConfig(**base)


def _pair(bs=2, t=16, vocab=64):
    return torch.randint(4, vocab, (bs, t)), torch.randint(4, vocab, (bs, t))


def _cache_equiv(model, src, tgt):
    """Tam decode ile self+cross cache'li kademeli decode ayni sonucu vermeli.

    SDPA cekirdekleri farkli q uzunluklarinda farkli toplama sirasi kullanabildigi
    icin float64'te karsilastirilir (sayisal gurultuyu eler)."""
    import copy

    m = copy.deepcopy(model).double().eval()
    with torch.no_grad():
        full = m(src, tgt)[:, -1, :]
        _, caches = m(src, tgt[:, :10], return_cache=True)
        out, _ = m(src, tgt[:, 10:], caches=caches, return_cache=True, pos_offset=10)
        return (out[:, -1, :] - full).abs().max().item()


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.src, self.tgt = _pair()

    def test_llama_style_forward_and_cache(self):
        model = EncoderDecoderLM(_tiny())
        self.assertEqual(model(self.src, self.tgt).shape, (2, 16, 64))
        self.assertLess(_cache_equiv(model, self.src, self.tgt), 1e-5)

    def test_gpt_style_layernorm_gelu_learned_pos(self):
        cfg = _tiny(norm_type="layernorm", activation="gelu", use_rope=False, bias=True)
        model = EncoderDecoderLM(cfg)
        self.assertIsInstance(model.pos_emb, torch.nn.Embedding)
        self.assertIsNone(model.rope)
        self.assertIsInstance(model.encoder[0].mlp, MLP)
        self.assertIsInstance(model.decoder[0].mlp, MLP)
        self.assertEqual(model(self.src, self.tgt).shape, (2, 16, 64))
        self.assertLess(_cache_equiv(model, self.src, self.tgt), 1e-5)

    def test_qk_norm_and_cache(self):
        model = EncoderDecoderLM(_tiny(qk_norm=True))
        self.assertEqual(model(self.src, self.tgt).shape, (2, 16, 64))
        self.assertLess(_cache_equiv(model, self.src, self.tgt), 1e-5)

    def test_linear_rope_scaling(self):
        cfg = _tiny(rope_scaling="linear", rope_scaling_factor=4.0)
        model = EncoderDecoderLM(cfg)
        self.assertAlmostEqual(model.rope.scaling_factor, 4.0)
        self.assertEqual(model(self.src, self.tgt).shape, (2, 16, 64))

    def test_z_loss_backward(self):
        cfg = _tiny(z_loss=0.01)
        model = EncoderDecoderLM(cfg)
        targets = torch.randint(0, 64, (2, 16))
        _, loss = model(self.src, self.tgt, targets=targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_pad_masks_keep_loss_finite(self):
        model = EncoderDecoderLM(_tiny())
        src_pad = torch.zeros(2, 16, dtype=torch.bool)
        src_pad[:, 12:] = True
        tgt_pad = torch.zeros(2, 16, dtype=torch.bool)
        tgt_pad[:, 10:] = True
        targets = torch.randint(0, 64, (2, 16))
        targets[:, 10:] = -100
        _, loss = model(self.src, self.tgt, targets=targets, src_pad=src_pad, tgt_pad=tgt_pad)
        self.assertTrue(torch.isfinite(loss))

    def test_factories(self):
        cfg = _tiny()
        self.assertIsInstance(make_norm(32, cfg), RMSNorm)
        self.assertIsInstance(make_ffn(cfg), SwiGLU)
        self.assertIsInstance(make_norm(32, _tiny(norm_type="layernorm")), torch.nn.LayerNorm)
        self.assertIsInstance(make_ffn(_tiny(activation="gelu")), MLP)

    def test_swiglu_hidden_rule(self):
        self.assertEqual(default_swiglu_hidden(256), 704)
        self.assertEqual(default_swiglu_hidden(32), 128)
        self.assertEqual(default_swiglu_hidden(256, multiple_of=1), 682)

    def test_residual_init_scaling_only_on_output_projections(self):
        cfg = _tiny(n_layers=8, d_model=64, n_head=4, ff_dim=256)
        model = EncoderDecoderLM(cfg)
        # Encoder blogu 2 residual dal -> 1/sqrt(2n); decoder blogu 3 dal -> 1/sqrt(3n).
        enc = 0.02 / (2 * 8) ** 0.5
        dec = 0.02 / (3 * 8) ** 0.5

        def std_of(name):
            return dict(model.named_parameters())[name].std().item()

        self.assertAlmostEqual(std_of("encoder.0.attn.proj.weight"), enc, delta=0.0015)
        self.assertAlmostEqual(std_of("encoder.0.mlp.down.weight"), enc, delta=0.0015)
        self.assertAlmostEqual(std_of("decoder.0.self_attn.proj.weight"), dec, delta=0.0015)
        self.assertAlmostEqual(std_of("decoder.0.cross_attn.proj.weight"), dec, delta=0.0015)
        self.assertAlmostEqual(std_of("decoder.0.mlp.down.weight"), dec, delta=0.0015)
        # q/k/v projeksiyonlari OLMAMALI (0.02 kalmali)
        self.assertAlmostEqual(std_of("decoder.0.self_attn.q_proj.weight"), 0.02, delta=0.002)
        self.assertAlmostEqual(std_of("decoder.0.self_attn.k_proj.weight"), 0.02, delta=0.002)
        self.assertAlmostEqual(std_of("decoder.0.self_attn.v_proj.weight"), 0.02, delta=0.002)

    def test_top_p_includes_boundary_token(self):
        import math

        logits = torch.tensor([[math.log(0.4), math.log(0.35), math.log(0.2), math.log(0.05)]])
        probs = EncoderDecoderLM._apply_sampler(logits, None, 0.7, None, "top_p", None, None)
        # 0.4 + 0.35 = 0.75 >= 0.7 -> esigi asan token dahil; 3. token (0.2) haric
        self.assertGreater(probs[0, 0].item(), 0.0)
        self.assertGreater(probs[0, 1].item(), 0.0)
        self.assertEqual(probs[0, 2].item(), 0.0)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_mirostat_v2_truncation(self):
        logits = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
        top1 = 3
        # mu buyuk -> hepsi kalir
        probs = EncoderDecoderLM._apply_sampler(logits, None, None, None, "mirostat", None, 20.0)
        self.assertEqual(int((probs > 0).sum().item()), 4)
        # mu kucuk -> yalniz en olasi token
        probs = EncoderDecoderLM._apply_sampler(logits, None, None, None, "mirostat", None, 5.0)
        self.assertEqual(int((probs > 0).sum().item()), 1)
        self.assertAlmostEqual(probs[0, top1].item(), 1.0, places=5)

    def test_prompt_history_order(self):
        from src.py.normal.prompt import ChatPromptFormatter

        fmt = ChatPromptFormatter(system_prompt="SYS")
        text = fmt.format_prompt("ikinci", history=[{"role": "user", "content": "birinci"},
                                                     {"role": "assistant", "content": "cevap"}])
        self.assertLess(text.index("birinci"), text.index("ikinci"))
        self.assertTrue(text.rstrip().endswith("assistant"))

    def test_tokenizer_prefix_space(self):
        from src.pt.decoder_model import BPETokenizer

        tok = BPETokenizer.from_corpus(
            ["merhaba dünya merhaba dünya merhaba dünya", "bugün hava çok güzel bugün hava çok güzel"],
            vocab_size=512,
        )
        # GPT-2 tarzi: bosluk, kelimeyle birlikte ogrenilen merge'lerde yer almali
        space_merges = [t for t in tok._id_to_token.values()
                        if isinstance(t, bytes) and len(t) > 1 and t[:1] == b" "]
        self.assertTrue(space_merges, "bosluk iceren merge ogrenilmedi")
        # ve metin kayipsiz cozulmeli
        self.assertEqual(tok.decode(tok.encode("merhaba dünya")), "merhaba dünya")


class ShimTests(unittest.TestCase):
    def test_new_symbols_reexported_by_decoder_model(self):
        from src.pt import decoder_model as dm

        for name in ("EncoderDecoderLM", "EncoderBlock", "DecoderBlock", "Attention",
                     "MLP", "make_norm", "make_ffn", "default_swiglu_hidden"):
            self.assertTrue(hasattr(dm, name), name)


if __name__ == "__main__":
    unittest.main()
