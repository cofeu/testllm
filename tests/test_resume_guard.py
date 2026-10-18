"""Resume guvenligi: tokenizer uyumsuzlugu ve eski .best mirasi.

Iki bulgu:
1) Resume her zaman checkpoint'in tokenizer'ini degil data/tokenizer.json'u
   kullanir; eski donem bir checkpoint + yeni tokenizer sessiz id kaymasina
   yol acar (vocab_size ayni oldugu icin hata da cikmaz).
2) Resume'suz yeni kosu, ayni --ckpt yolundaki eski .best'in best_val'ini
   okur; ilk eval onu gecemeyince yeni kosunun .best'i hic yazilmaz.
"""

import os
import tempfile
import unittest

import torch

from src.py.normal import BPETokenizer, TrainConfig, save_checkpoint
from src.py.normal.train import (
    ensure_checkpoint_tokenizer,
    initial_best_state,
    tokenizer_identity,
    train_model,
)


def _texts(n=30):
    return [{"user": f"soru {i}", "assistant": f"cevap {i} burada"} for i in range(n)]


def _tok(texts):
    return BPETokenizer.from_corpus(texts, vocab_size=300)


class ResumeTokenizerGuardTests(unittest.TestCase):
    def test_matching_tokenizer_passes(self):
        tok = _tok(["merhaba dunya", "cevap"])
        ensure_checkpoint_tokenizer(tok.to_dict(), tok, "x.pt")

    def test_missing_checkpoint_tokenizer_is_skipped(self):
        tok = _tok(["merhaba"])
        ensure_checkpoint_tokenizer(None, tok, "x.pt")
        ensure_checkpoint_tokenizer(tok.to_dict(), None, "x.pt")

    def test_v2_era_checkpoint_rejected(self):
        # v2 dosyasi: offset alani yok, 6 ozel token, surum 2.
        tok = _tok(["merhaba dunya", "cevap"])
        old = tok.to_dict()
        old["version"] = 2
        old["specials"] = old["specials"][:6]
        del old["offset"]
        with self.assertRaises(ValueError) as ctx:
            ensure_checkpoint_tokenizer(old, tok, "old.pt")
        msg = str(ctx.exception)
        self.assertIn("resume", msg)
        self.assertIn("offset", msg)

    def test_identity_reports_expected_fields(self):
        tok = _tok(["merhaba"])
        ident = tokenizer_identity(tok.to_dict())
        self.assertEqual(ident["offset"], BPETokenizer.OFFSET)
        self.assertEqual(ident["specials"], list(BPETokenizer.SPECIAL_STRINGS))
        self.assertEqual(ident["version"], BPETokenizer.VERSION)

    def test_resume_path_actually_checks(self):
        # train_model resume kolu gercekten kontrol ediyor mu? (integre test)
        texts = _texts()
        tok_a = _tok([t["user"] for t in texts] + [t["assistant"] for t in texts])
        cfg = TrainConfig(vocab_size=tok_a.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=64, batch_size=4, epochs=2, eval_every=1,
                          save_every=1, log_every=1, resume=True, amp=False)
        model = torch.nn.Module()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            from src.py.normal import EncoderDecoderLM

            m = EncoderDecoderLM(cfg)
            save_checkpoint(m, cfg, path, tokenizer=tok_a, step=1)

            # Checkpoint'i v2 donemine cevir (offset/specials/surum).
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["tokenizer"]["version"] = 2
            payload["tokenizer"]["specials"] = payload["tokenizer"]["specials"][:6]
            del payload["tokenizer"]["offset"]
            torch.save(payload, path)

            tok_b = _tok([t["user"] for t in texts] + ["farkli metin"])
            with self.assertRaises(ValueError):
                train_model(cfg, device=torch.device("cpu"), ckpt_path=path,
                            tokenizer=tok_b, texts=texts)


class BestInheritanceTests(unittest.TestCase):
    def _write_best(self, path, best_val):
        torch.save({"meta": {"best_val": best_val}}, path)

    def test_resume_inherits_old_best(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt.best")
            self._write_best(path, 1.5)
            cfg = TrainConfig(vocab_size=64, resume=True)
            self.assertEqual(initial_best_state(cfg, path, None), 1.5)

    def test_fresh_run_ignores_old_best(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt.best")
            self._write_best(path, 1.5)
            cfg = TrainConfig(vocab_size=64, resume=False)
            self.assertEqual(initial_best_state(cfg, path, None), float("inf"))

    def test_fresh_run_writes_its_own_best(self):
        # Eski .best ulasilamaz bir best_val ile kirli birakilmis olsa bile
        # resume'suz kosu ilk eval'da kendi .best'ini yazmali.
        texts = _texts()
        tok = _tok([t["user"] for t in texts] + [t["assistant"] for t in texts])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=64, batch_size=4, epochs=2, eval_every=1,
                          save_every=1, log_every=1, resume=False,
                          amp=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            self._write_best(path + ".best", 1e-6)  # eski kosunun "ulasilamaz" best'i
            before = torch.load(path + ".best", map_location="cpu", weights_only=True)
            self.assertEqual(before["meta"]["best_val"], 1e-6)

            train_model(cfg, device=torch.device("cpu"), ckpt_path=path, tokenizer=tok,
                        texts=texts, keep_best=True)

            after = torch.load(path + ".best", map_location="cpu", weights_only=True)
            self.assertNotEqual(after["meta"]["best_val"], 1e-6)
            self.assertGreater(after["meta"]["best_val"], 1e-6)

    def test_resume_keeps_old_best(self):
        texts = _texts()
        tok = _tok([t["user"] for t in texts] + [t["assistant"] for t in texts])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=64, batch_size=4, epochs=2, eval_every=1,
                          save_every=1, log_every=1, resume=True,
                          amp=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            from src.py.normal import EncoderDecoderLM

            save_checkpoint(EncoderDecoderLM(cfg), cfg, path, tokenizer=tok, step=1,
                            meta={"best_val": 0.5})
            self._write_best(path + ".best", 0.5)

            train_model(cfg, device=torch.device("cpu"), ckpt_path=path, tokenizer=tok,
                        texts=texts, keep_best=True)

            after = torch.load(path + ".best", map_location="cpu", weights_only=True)
            # 0.5 ulasilamaz; miras alinan best_val korunmali (ezilmemeli).
            self.assertEqual(after["meta"]["best_val"], 0.5)


if __name__ == "__main__":
    unittest.main()
