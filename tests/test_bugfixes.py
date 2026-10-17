import json
import os
import tempfile
import unittest

import torch

from src.py.normal import (
    BPETokenizer,
    InstructionDataset,
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
        from src.py.normal import DecoderOnlyLM

        path = tempfile.mktemp(suffix=".pt")
        try:
            save_checkpoint(DecoderOnlyLM(cfg), cfg, path, step=3, tokenizer=tok)
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

    def test_pack_flag_controls_dataset_mode(self):
        texts = _small_texts()
        tok = _tok([t["user"] for t in texts] + [t["assistant"] for t in texts])
        base = dict(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                    ff_dim=0, block_size=64, dropout=0.0, batch_size=4)

        packed = InstructionDataset(tok, TrainConfig(**base, pack_sequences=True), texts)
        per = InstructionDataset(tok, TrainConfig(**base, pack_sequences=False), texts)

        self.assertEqual(packed.mode, "packed")
        self.assertEqual(per.mode, "per_example")
        # per_example: her ornek ayri satir, sinir asilmaz
        self.assertEqual(per.n_train + per.n_val, per.n_examples)
        y = per.get_batch("train", 4, torch.device("cpu"))[1]
        self.assertGreater(int((y != per.IGNORE).sum()), 0)

    def test_z_loss_ignores_all_prompt_windows(self):
        tok = _tok(["merhaba dunya"])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=8, dropout=0.0, z_loss=0.01)
        from src.py.normal import DecoderOnlyLM

        model = DecoderOnlyLM(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.full((2, 8), -100)
        _, loss = model(x, targets=y)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)
        loss.backward()

    def test_ff_dim_auto_uses_llama_rule(self):
        from src.py.normal.model import default_swiglu_hidden

        cfg = TrainConfig(vocab_size=64, d_model=32, n_layers=1, n_head=4, ff_dim=0)
        from src.py.normal import DecoderOnlyLM

        model = DecoderOnlyLM(cfg)
        self.assertEqual(model.blocks[0].mlp.hidden, default_swiglu_hidden(32))

    def test_resume_seed_offset_changes_batch_stream(self):
        from src.py.normal import seed_everything

        texts = _small_texts()
        tok = _tok([t["user"] for t in texts] + [t["assistant"] for t in texts])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=64, batch_size=4)
        ds = InstructionDataset(tok, cfg, texts)

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
