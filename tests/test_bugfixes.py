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
        # Dinamik pad: batch'teki en uzun diziye kadar (block_size'i asmaz)
        self.assertEqual(src.size(0), 4)
        self.assertLessEqual(src.size(1), cfg.block_size)
        self.assertEqual(tgt_in.shape, tgt_out.shape)
        self.assertLessEqual(tgt_in.size(1), cfg.block_size)
        # decoder girdisi BOS ile baslar; gercek cevap token'lari var
        self.assertEqual(int(tgt_in[0, 0]), tok.BOS)
        self.assertGreater(int((tgt_out != ds.IGNORE).sum()), 0)

    def test_teacher_forcing_alignment_and_single_eos(self):
        texts = _small_texts(20)
        tok = _tok([t["user"] for t in texts] + [t["assistant"] for t in texts])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=64, dropout=0.0, batch_size=4)
        ds = Seq2SeqDataset(tok, cfg, texts)
        _, tgt_in, tgt_out = ds.get_batch("train", 4, torch.device("cpu"))
        for r in range(4):
            row_in = tgt_in[r][tgt_in[r] != tok.PAD]
            row_out = tgt_out[r][tgt_out[r] != ds.IGNORE]
            self.assertEqual(int(row_in[0]), tok.BOS)
            self.assertEqual(int(row_out[-1]), tok.IM_END)
            # tgt_in = [BOS] + answer ; tgt_out = answer + [IM_END]
            self.assertTrue(torch.equal(row_in[1:], row_out[:-1]))
            # IM_END tam olarak BIR kez (cift EOS yok)
            self.assertEqual(int((row_out == tok.IM_END).sum()), 1)

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

    def test_length_bucketing_groups_similar_lengths(self):
        texts = _small_texts(8)
        tok = _tok([t["user"] for t in texts] + [t["assistant"] for t in texts])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=200, batch_size=4, val_frac=0.25)
        ds = Seq2SeqDataset(tok, cfg, texts)

        # [5,6,7,100,101,20] token'lik satirlar; 8'lik kovalar:
        # 5,6,7 -> kova 0 ; 20 -> kova 2 ; 100,101 -> kova 12
        lengths = [5, 6, 7, 100, 101, 20]
        _, off_s = Seq2SeqDataset._flat([[0] * l for l in lengths])
        _, off_o = Seq2SeqDataset._flat([[0] * l for l in lengths])
        _, bucket_of = ds._bucketize(off_s, off_o)

        self.assertEqual(bucket_of[0], bucket_of[1])
        self.assertEqual(bucket_of[1], bucket_of[2])
        self.assertEqual(bucket_of[3], bucket_of[4])
        self.assertNotEqual(bucket_of[0], bucket_of[3])
        self.assertNotEqual(bucket_of[5], bucket_of[3])

    def test_truncated_answer_has_no_eos(self):
        texts = [{"user": "kisa soru", "assistant": "kelime " * 80}]
        tok = _tok([t["user"] for t in texts] + [t["assistant"] for t in texts])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=16, dropout=0.0, batch_size=1)
        ds = Seq2SeqDataset(tok, cfg, texts)
        _, tgt_in, tgt_out = ds.get_batch("train", 1, torch.device("cpu"))
        row_out = tgt_out[0][tgt_out[0] != ds.IGNORE]
        # Kirpilan cevaba EOS eklenmez (yarida bitirmeyi ogrenmesin).
        self.assertEqual(int((row_out == tok.IM_END).sum()), 0)
        self.assertEqual(row_out.numel(), cfg.block_size - 1)

    def test_generate_accepts_pad_id(self):
        tok = _tok(["merhaba dunya", "nasilsin"])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=32)
        model = EncoderDecoderLM(cfg)
        src = torch.tensor([[tok.BOS] + tok.encode("merhaba") + [tok.PAD, tok.PAD]])
        out = model.generate(src, max_new_tokens=3, pad_id=tok.PAD, bos_id=tok.BOS,
                             eos_id=tok.EOS, device=torch.device("cpu"))
        self.assertEqual(int(out[0, 0]), tok.BOS)
        self.assertGreaterEqual(out.size(1), 1)

    def test_multiturn_history_in_corpus_and_prompt(self):
        from src.py.normal.corpus import build_assistant_only_corpus
        from src.py.normal.prompt import ChatPromptFormatter

        dialogues = [[["ilk soru", "ilk cevap burada"], ["ikinci soru", "ikinci cevap burada"]]]
        corpus = build_assistant_only_corpus(dialogues)
        self.assertEqual(len(corpus), 2)
        # Ilk turun gecmisi bos, ikinci turun gecmisi ilk turu icerir.
        self.assertEqual(corpus[0]["history"], [])
        self.assertEqual([m["content"] for m in corpus[1]["history"]], ["ilk soru", "ilk cevap burada"])

        prompt = ChatPromptFormatter().format_prompt(
            corpus[1]["user"], history=corpus[1]["history"])
        # Gecmis, guncel kullanici mesajindan ONCE ve dogru sirada yer alir.
        self.assertIn("ilk soru", prompt)
        self.assertLess(prompt.index("ilk soru"), prompt.index("ikinci soru"))
        self.assertTrue(prompt.rstrip().endswith("<|im_start|>assistant"))

    def test_multiturn_history_lengthens_dataset_source(self):
        texts = [{"user": "ikinci soru", "assistant": "ikinci cevap burada", "system": "",
                  "history": [{"role": "user", "content": "ilk soru burada"},
                              {"role": "assistant", "content": "ilk cevap burada"}]}]
        plain = [{"user": "ikinci soru", "assistant": "ikinci cevap burada", "system": ""}]
        tok = _tok(["ilk soru burada", "ilk cevap burada", "ikinci soru", "ikinci cevap burada"])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=128, batch_size=1)
        with_hist = Seq2SeqDataset(tok, cfg, texts).get_batch("train", 1, torch.device("cpu"))[0]
        without = Seq2SeqDataset(tok, cfg, plain).get_batch("train", 1, torch.device("cpu"))[0]
        self.assertGreater(int((with_hist != tok.PAD).sum()), int((without != tok.PAD).sum()))

    def test_clean_text_keep_newlines(self):
        from src.py.normal.prompt import clean_text

        raw = "Adım 1:  bir   şey\n\n\nAdım 2:\tiki\n"
        flat = clean_text(raw)
        self.assertNotIn("\n", flat)
        kept = clean_text(raw, keep_newlines=True)
        self.assertIn("\n", kept)
        self.assertIn("Adım 1:", kept)
        self.assertIn("Adım 2:", kept)
        self.assertNotIn("\n\n\n", kept)
        # Ayni satir icindeki fazla bosluk yine sadelesir.
        self.assertIn("Adım 1: bir şey", kept)

    def test_messages_to_turns(self):
        from src.py.normal.corpus import messages_to_turns

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "birinci"},
            {"role": "assistant", "content": "cevap bir"},
            {"role": "user", "content": "ikinci"},
            {"role": "user", "content": "devam"},
            {"role": "assistant", "content": "cevap iki"},
            {"role": "user", "content": "cevapsiz"},
        ]
        turns = messages_to_turns(messages)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0], ["birinci", "cevap bir"])
        self.assertEqual(turns[1], ["ikinci\ndevam", "cevap iki"])

    def test_load_jsonl_reads_messages(self):
        from src.py.normal.corpus import load_jsonl_dialogues

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "chatml.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": "selam"},
                    {"role": "assistant", "content": "merhaba"},
                ]}, ensure_ascii=False) + "\n")
            rows = load_jsonl_dialogues(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0][0], "selam")
        self.assertEqual(rows[0][0][1], "merhaba")

    def test_max_examples_applied_after_filtering(self):
        # Ilk ornekler elenir (kisa); max_examples eleme SONRASI uygulanmali.
        dialogues = [[["ab", "x"], ["gecerli soru", f"gecerli cevap {i} burada"]] for i in range(5)]
        corpus = build_assistant_only_corpus(dialogues, max_examples=4)
        self.assertEqual(len(corpus), 4)
        self.assertTrue(all(len(c["assistant"]) >= 4 for c in corpus))

    def test_corpus_preserves_multiline_answer(self):
        dialogues = [[["adım adım", "Adım 1: düşün\nAdım 2: uygula\nSonuç: tamam"]]]
        corpus = build_assistant_only_corpus(dialogues)
        self.assertEqual(len(corpus), 1)
        self.assertIn("\n", corpus[0]["assistant"])
        self.assertIn("Adım 1", corpus[0]["assistant"])

    def test_generate_unk_id_param(self):
        tok = _tok(["merhaba dunya"])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=16)
        model = EncoderDecoderLM(cfg)
        src = torch.tensor([[tok.BOS] + tok.encode("merhaba")])
        out = model.generate(src, max_new_tokens=2, pad_id=tok.PAD, bos_id=tok.BOS,
                             eos_id=tok.EOS, unk_id=tok.UNK, device=torch.device("cpu"))
        gen = out[0].tolist()
        # UNK uretilmemeli (forbidden).
        self.assertNotIn(tok.UNK, gen)

    def test_encoder_self_attention_bidirectional(self):
        # Encoder cift yonlu olmali: SON kaynak token'i degisince ILK encoder
        # konumunun temsili de degismeli. (Eski hata: encoder'a yanlislikla
        # nedensel maske uygulaniyordu.)
        tok = _tok(["kelime"] * 8)
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=16, dropout=0.0)
        model = EncoderDecoderLM(cfg).eval()
        ids = [tok.OFFSET + 1, tok.OFFSET + 2, tok.OFFSET + 3, tok.OFFSET + 4]
        src1 = torch.tensor([ids])
        src2 = torch.tensor([ids[:3] + [tok.OFFSET + 5]])
        with torch.no_grad():
            m1 = model.encode(src1)
            m2 = model.encode(src2)
        self.assertFalse(torch.allclose(m1[0, 0], m2[0, 0], atol=1e-6))

    def test_decoder_self_attention_causal(self):
        # Decoder nedensel kalmali: hedefin SON token'i degisince ILK konumun
        # logit'i degismemeli.
        tok = _tok(["kelime"] * 8)
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=16, dropout=0.0)
        model = EncoderDecoderLM(cfg).eval()
        src = torch.tensor([[tok.OFFSET + 1, tok.OFFSET + 2, tok.OFFSET + 3]])
        ids = [tok.BOS, tok.OFFSET + 4, tok.OFFSET + 5, tok.OFFSET + 6]
        tgt1 = torch.tensor([ids])
        tgt2 = torch.tensor([ids[:3] + [tok.OFFSET + 7]])
        with torch.no_grad():
            a = model(src, tgt1)
            b = model(src, tgt2)
        self.assertTrue(torch.allclose(a[0, 0], b[0, 0], atol=1e-5))
        self.assertFalse(torch.allclose(a[0, -1], b[0, -1]))


if __name__ == "__main__":
    unittest.main()
