import unittest

import torch

from src.pt.decoder_model import DecoderOnlyLM, TrainConfig


class CausalArchitectureTests(unittest.TestCase):
    def _model(self):
        cfg = TrainConfig(
            vocab_size=64,
            d_model=32,
            n_layers=2,
            n_head=4,
            ff_dim=64,
            block_size=16,
            dropout=0.0,
            seed=7,
        )
        return cfg, DecoderOnlyLM(cfg).eval()

    def test_is_causal_only(self):
        cfg, model = self._model()
        self.assertFalse(hasattr(model, "encoder_layers"))
        self.assertGreaterEqual(cfg.n_layers, 2)

    def test_future_tokens_do_not_affect_past(self):
        cfg, model = self._model()
        torch.manual_seed(0)
        ids = torch.randint(4, cfg.vocab_size, (1, 10))
        with torch.no_grad():
            logits_a = model(ids)
            shifted = ids.clone()
            shifted[0, -1] = (shifted[0, -1] + 7) % cfg.vocab_size
            logits_b = model(shifted)

        self.assertTrue(torch.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5))
        self.assertFalse(torch.allclose(logits_a[0, -1], logits_b[0, -1]))


if __name__ == "__main__":
    unittest.main()
