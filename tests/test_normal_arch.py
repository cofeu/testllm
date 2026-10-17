import unittest

import torch

from src.py.normal import (
    DecoderOnlyLM,
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


def _cache_equiv(model, tokens):
    model.eval()
    with torch.no_grad():
        full = model(tokens)[:, -1, :]
        _, caches = model(tokens[:, :10], return_cache=True)
        out, _ = model(tokens[:, 10:], caches=caches, return_cache=True, pos_offset=10)
        return (out[:, -1, :] - full).abs().max().item()


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.tokens = torch.randint(0, 64, (2, 16))

    def test_llama_style_forward_and_cache(self):
        model = DecoderOnlyLM(_tiny())
        self.assertEqual(model(self.tokens).shape, (2, 16, 64))
        self.assertLess(_cache_equiv(model, self.tokens), 1e-5)

    def test_gpt_style_layernorm_gelu_learned_pos(self):
        cfg = _tiny(norm_type="layernorm", activation="gelu", use_rope=False, bias=True)
        model = DecoderOnlyLM(cfg)
        self.assertIsInstance(model.pos_emb, torch.nn.Embedding)
        self.assertIsNone(model.rope)
        self.assertIsInstance(model.blocks[0].mlp, MLP)
        self.assertEqual(model(self.tokens).shape, (2, 16, 64))
        self.assertLess(_cache_equiv(model, self.tokens), 1e-5)

    def test_qk_norm_and_cache(self):
        model = DecoderOnlyLM(_tiny(qk_norm=True))
        self.assertEqual(model(self.tokens).shape, (2, 16, 64))
        self.assertLess(_cache_equiv(model, self.tokens), 1e-5)

    def test_linear_rope_scaling(self):
        cfg = _tiny(rope_scaling="linear", rope_scaling_factor=4.0)
        model = DecoderOnlyLM(cfg)
        self.assertAlmostEqual(model.rope.scaling_factor, 4.0)
        self.assertEqual(model(self.tokens).shape, (2, 16, 64))

    def test_z_loss_backward(self):
        cfg = _tiny(z_loss=0.01)
        model = DecoderOnlyLM(cfg)
        targets = torch.randint(0, 64, (2, 16))
        _, loss = model(self.tokens, targets=targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_gqa_and_loss_mask(self):
        model = DecoderOnlyLM(_tiny())
        targets = torch.randint(0, 64, (2, 16))
        mask = torch.ones(2, 16)
        mask[:, :8] = 0.0
        _, loss = model(self.tokens, targets=targets, loss_mask=mask)
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
        model = DecoderOnlyLM(cfg)
        expected = 0.02 / (2 * 8) ** 0.5

        def std_of(name):
            return dict(model.named_parameters())[name].std().item()

        # Cikis projeksiyonlari olceklenmeli
        self.assertAlmostEqual(std_of("blocks.0.attn.proj.weight"), expected, delta=0.0015)
        self.assertAlmostEqual(std_of("blocks.0.mlp.down.weight"), expected, delta=0.0015)
        # q/k/v projeksiyonlari OLMAMALI (0.02 kalmali)
        self.assertAlmostEqual(std_of("blocks.0.attn.q_proj.weight"), 0.02, delta=0.002)
        self.assertAlmostEqual(std_of("blocks.0.attn.k_proj.weight"), 0.02, delta=0.002)
        self.assertAlmostEqual(std_of("blocks.0.attn.v_proj.weight"), 0.02, delta=0.002)

    def test_mirostat_v2_truncation(self):
        logits = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
        top1 = 3
        # mu buyuk -> hepsi kalir
        probs = DecoderOnlyLM._apply_sampler(logits, None, None, None, "mirostat", None, 20.0)
        self.assertEqual(int((probs > 0).sum().item()), 4)
        # mu kucuk -> yalniz en olasi token
        probs = DecoderOnlyLM._apply_sampler(logits, None, None, None, "mirostat", None, 5.0)
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

        for name in ("MLP", "make_norm", "make_ffn", "default_swiglu_hidden"):
            self.assertTrue(hasattr(dm, name), name)


if __name__ == "__main__":
    unittest.main()
