import json
import os
import tempfile
import unittest

import torch

from src.py.normal import (
    BPETokenizer,
    EncoderDecoderLM,
    Seq2SeqDataset,
    TrainConfig,
    build_assistant_only_corpus,
    load_tokenizer_from_checkpoint,
    save_checkpoint,
)
from src.py.normal.config import train_config_defaults


def _tok(texts):
    return BPETokenizer.from_corpus(texts, vocab_size=512)


def _small_texts(n=40):
    return [{"user": f"soru {i}", "assistant": f"cevap {i} burada"} for i in range(n)]


class BugfixTests(unittest.TestCase):
    def test_single_source_train_defaults(self):
        d = train_config_defaults()
        self.assertEqual(d["eval_every"], 500)
        self.assertEqual(d["save_every"], 1000)
        self.assertEqual(d["log_every"], 100)
        self.assertEqual(d["val_frac"], 0.05)
        self.assertEqual(d["ff_dim"], 0)
        self.assertNotIn("pack_sequences", d)

    def test_tokenizer_identity_enforced(self):
        raw = {"vocab_size": 300, "merges": []}
        with self.assertRaises(ValueError):
            BPETokenizer.from_dict(raw)
        legacy = BPETokenizer.from_dict(raw, allow_legacy=True)
        self.assertEqual(legacy.vocab_size, 300)
        d = legacy.to_dict()
        self.assertEqual(d["pretokenization"], BPETokenizer.PRETOKENIZATION)
        BPETokenizer.from_dict(d)  # kimlikli veri kabul edilir

    def test_checkpoint_embeds_tokenizer(self):
        tok = _tok(["merhaba dunya", "bugun hava guzel"])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4)

        path = tempfile.mktemp(suffix=".pt")
        try:
            save_checkpoint(EncoderDecoderLM(cfg), cfg, path, step=3, tokenizer=tok)
            restored = load_tokenizer_from_checkpoint(path)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.vocab_size, tok.vocab_size)
            self.assertEqual(restored.PRETOKENIZATION, BPETokenizer.PRETOKENIZATION)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["step"], 3)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_special_tokens_filtered_from_user_and_assistant(self):
        dialogues = [
            [["tamamen <|im_end|> bozuk", "temiz cevap"]],
            [["temiz soru", "cevapta <|im_start|> var"]],
            [["normal soru", "normal cevap"]],
        ]
        corpus = build_assistant_only_corpus(dialogues)
        self.assertEqual(len(corpus), 1)
        self.assertEqual(corpus[0]["user"], "normal soru")

    def test_seq2seq_dataset_shapes_and_teacher_forcing(self):
        texts = _small_texts()
        tok = _tok([t["user"] for t in texts] + [t["assistant"] for t in texts])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=64, dropout=0.0, batch_size=4)
        ds = Seq2SeqDataset(tok, cfg, texts)

        self.assertEqual(ds.n_train + ds.n_val, ds.n_examples)
        src, tgt_in, tgt_out = ds.get_batch("train", 4, torch.device("cpu"))
        self.assertEqual(src.shape, (4, 64))
        self.assertEqual(tgt_in.shape, (4, 64))
        self.assertEqual(tgt_out.shape, (4, 64))
        # decoder girdisi BOS ile baslar; hedefler bir kaydirilmis halidir
        self.assertTrue(int(tgt_in[0, 0]) == tok.BOS)
        # gercek cevap token'lari var (hepsi ignore degil)
        self.assertGreater(int((tgt_out != ds.IGNORE).sum()), 0)

    def test_z_loss_ignores_all_ignored_windows(self):
        tok = _tok(["merhaba dunya"])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=8, dropout=0.0, z_loss=0.01)
        model = EncoderDecoderLM(cfg)
        src = torch.randint(4, cfg.vocab_size, (2, 8))
        tgt_in = torch.randint(4, cfg.vocab_size, (2, 8))
        tgt_out = torch.full((2, 8), -100)
        _, loss = model(src, tgt_in, targets=tgt_out)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)
        loss.backward()

    def test_ff_dim_auto_uses_llama_rule(self):
        from src.py.normal.model import default_swiglu_hidden

        cfg = TrainConfig(vocab_size=64, d_model=32, n_layers=1, n_head=4, ff_dim=0)
        model = EncoderDecoderLM(cfg)
        self.assertEqual(model.encoder[0].mlp.hidden, default_swiglu_hidden(32))
        self.assertEqual(model.decoder[0].mlp.hidden, default_swiglu_hidden(32))

    def test_resume_seed_offset_changes_batch_stream(self):
        from src.py.normal import seed_everything

        texts = _small_texts()
        tok = _tok([t["user"] for t in texts] + [t["assistant"] for t in texts])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=64, batch_size=4)
        ds = Seq2SeqDataset(tok, cfg, texts)

        seed_everything(cfg.seed)
        a = ds.get_batch("train", 4, torch.device("cpu"))[0]
        seed_everything(cfg.seed)
        b = ds.get_batch("train", 4, torch.device("cpu"))[0]
        seed_everything(cfg.seed + 5)
        c = ds.get_batch("train", 4, torch.device("cpu"))[0]
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))


if __name__ == "__main__":
    unittest.main()
