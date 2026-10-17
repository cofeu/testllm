import unittest

import torch

from src.pt.decoder_model import EncoderDecoderLM, TrainConfig
from src.py.normal.model import Attention


class EncoderDecoderTests(unittest.TestCase):
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
        return cfg, EncoderDecoderLM(cfg).eval()

    def test_has_encoder_decoder_and_cross_attention(self):
        cfg, model = self._model()
        self.assertEqual(len(model.encoder), cfg.n_layers)
        self.assertEqual(len(model.decoder), cfg.n_layers)
        self.assertIsInstance(model.decoder[0].cross_attn, Attention)
        self.assertTrue(model.decoder[0].cross_attn.cross)

    def test_encoder_is_bidirectional(self):
        """Encoder cift yonlu: kaynagin SON token'i degisince decoder ilk
        konumunun logit'leri de degismeli (cross-attention tum kaynagi gorur)."""
        cfg, model = self._model()
        torch.manual_seed(0)
        src = torch.randint(4, cfg.vocab_size, (1, 10))
        tgt = torch.randint(4, cfg.vocab_size, (1, 10))
        with torch.no_grad():
            a = model(src, tgt)
            shifted = src.clone()
            shifted[0, -1] = (shifted[0, -1] + 7) % cfg.vocab_size
            b = model(shifted, tgt)
        self.assertFalse(torch.allclose(a[0, 0], b[0, 0]))

    def test_decoder_is_causal(self):
        """Decoder nedensel: hedefin SON token'i degisince onceki konumlar
        etkilenmemeli."""
        cfg, model = self._model()
        torch.manual_seed(0)
        src = torch.randint(4, cfg.vocab_size, (1, 10))
        tgt = torch.randint(4, cfg.vocab_size, (1, 10))
        with torch.no_grad():
            a = model(src, tgt)
            shifted = tgt.clone()
            shifted[0, -1] = (shifted[0, -1] + 7) % cfg.vocab_size
            b = model(src, shifted)
        self.assertTrue(torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5))
        self.assertFalse(torch.allclose(a[0, -1], b[0, -1]))


if __name__ == "__main__":
    unittest.main()
