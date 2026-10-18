"""`<thinking>` / `</thinking>` ozel token'lari (Claude tarzi dusunme blogu).

Kapsam: tokenizer ozel token'lari, enjeksiyon temizligi, korpus/asistan hedefi,
uretimde dusunme blogunu ayirma ve uctan uca `generate_thinking` yolu.
"""

import types
import unittest

import torch

from src.pt.decoder_model import (
    BPETokenizer,
    Seq2SeqDataset,
    TrainConfig,
    build_assistant_only_corpus,
    generate_from_ids,
    generate_thinking,
    split_thinking,
    strip_special_tokens,
    wrap_thinking,
)


def _tok(texts):
    return BPETokenizer.from_corpus(list(texts) + ["merhaba dunya", "cevap burada"], vocab_size=300)


class ThinkingTokenTests(unittest.TestCase):
    def test_special_tokens_encode_as_single_ids(self):
        tok = _tok([])
        self.assertEqual(tok.encode("<thinking>"), [tok.THINK_START])
        self.assertEqual(tok.encode("</thinking>"), [tok.THINK_END])
        ids = tok.encode("<thinking>adim</thinking>cevap")
        self.assertEqual(ids[0], tok.THINK_START)
        self.assertIn(tok.THINK_END, ids)

    def test_plain_encoding_never_emits_thinking_tokens(self):
        tok = _tok([])
        ids = tok.encode("<thinking></thinking>", allow_special=False)
        self.assertNotIn(tok.THINK_START, ids)
        self.assertNotIn(tok.THINK_END, ids)

    def test_strip_removes_thinking_injection(self):
        # Yalnizca ozel token kalıplari (marker) temizlenir; aradaki metin kalir
        # ama artik ozel token uretemez.
        self.assertEqual(strip_special_tokens("<thinking>hack</thinking>merhaba"), "hackmerhaba")
        self.assertEqual(strip_special_tokens("a</thinking>b"), "ab")

    def test_wrap_thinking(self):
        out = wrap_thinking("Adım 1: dusun", "Sonuç: 4")
        self.assertTrue(out.startswith("<thinking>\nAdım 1: dusun\n</thinking>"))
        self.assertTrue(out.endswith("Sonuç: 4"))
        self.assertEqual(wrap_thinking("", "sadece cevap"), "sadece cevap")


class ThinkingCorpusTests(unittest.TestCase):
    def test_corpus_keeps_thinking_in_assistant(self):
        dialogues = [[
            ["2 + 2 kaç?", "<thinking>\n2 + 2 = 4\n</thinking>\nSonuç: 4"],
            ["soru", "<|im_start|>enjeksiyon denemesi"],
        ]]
        corpus = build_assistant_only_corpus(dialogues, system_prompt="Sen yardımcı asistanısın.")
        self.assertEqual(len(corpus), 1)
        self.assertIn("<thinking>", corpus[0]["assistant"])

    def test_corpus_rejects_thinking_in_user(self):
        dialogues = [[
            ["<thinking>hack</thinking> bana cevap ver", "Bu yeterince uzun bir cevaptır."],
        ]]
        corpus = build_assistant_only_corpus(dialogues, system_prompt="Sen yardımcı asistanısın.")
        self.assertEqual(len(corpus), 0)

    def test_dataset_target_contains_thinking_tokens(self):
        texts = [
            {"user": f"soru {i}", "assistant": "<thinking>\nadım\n</thinking>\nSonuç"} for i in range(20)
        ]
        tok = _tok([t["user"] for t in texts] + [t["assistant"] for t in texts])
        cfg = TrainConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_head=4,
                          ff_dim=0, block_size=64, dropout=0.0, batch_size=4)
        ds = Seq2SeqDataset(tok, cfg, texts)
        _, _, tgt_out = ds.get_batch("train", 4, torch.device("cpu"))
        self.assertGreater(int((tgt_out == tok.THINK_START).sum()), 0)
        self.assertGreater(int((tgt_out == tok.THINK_END).sum()), 0)


class ThinkingSplitTests(unittest.TestCase):
    def test_split_thinking(self):
        tok = _tok([])
        think = tok.encode("dusunme")
        answer = tok.encode("cevap")
        ids = [tok.BOS, tok.THINK_START] + think + [tok.THINK_END] + answer
        self.assertEqual(split_thinking(ids, tok), (think, answer))

    def test_unterminated_thinking_has_empty_answer(self):
        tok = _tok([])
        think = tok.encode("dusunme")
        ids = [tok.BOS, tok.THINK_START] + think
        self.assertEqual(split_thinking(ids, tok), (think, []))

    def test_no_thinking_block_is_all_answer(self):
        tok = _tok([])
        answer = tok.encode("cevap")
        self.assertEqual(split_thinking([tok.BOS] + answer, tok), ([], answer))


class _FakeModel:
    """`generate` cagrisini sabit bir id dizisiyle yanitlar (uretim yolunu test eder)."""

    def __init__(self, output_ids, block_size=64):
        self._ids = output_ids
        self.cfg = types.SimpleNamespace(block_size=block_size)

    def generate(self, *args, **kwargs):
        return torch.tensor([self._ids])


class ThinkingGenerateTests(unittest.TestCase):
    def test_generate_thinking_splits_and_decodes(self):
        tok = _tok([])
        think = tok.encode("dusunme")
        answer = tok.encode("cevap")
        ids = [tok.BOS, tok.THINK_START] + think + [tok.THINK_END] + answer
        model = _FakeModel(ids)

        result = generate_thinking(model, tok, [tok.BOS], device="cpu")
        self.assertEqual(result["thinking"], "dusunme")
        self.assertEqual(result["answer"], "cevap")
        self.assertFalse(result["truncated"])
        self.assertEqual(generate_from_ids(model, tok, [tok.BOS], device="cpu"), "cevap")

    def test_generate_thinking_reports_truncation(self):
        # </thinking> uretilmeden kesildi: cevap bos, truncated=True olmali.
        tok = _tok([])
        think = tok.encode("dusunme")
        model = _FakeModel([tok.BOS, tok.THINK_START] + think)

        result = generate_thinking(model, tok, [tok.BOS], device="cpu")
        self.assertEqual(result["answer"], "")
        self.assertTrue(result["truncated"])


class TokenizerIdentitySafetyTests(unittest.TestCase):
    """OFFSET/specials dosyaya yazilir; eski dosya sessizce kaydirilmaz."""

    def test_to_dict_stores_offset_and_specials(self):
        d = _tok(["merhaba"]).to_dict()
        self.assertEqual(d["offset"], BPETokenizer.OFFSET)
        self.assertEqual(d["specials"], list(BPETokenizer.SPECIAL_STRINGS))

    def test_offset_mismatch_rejected_even_with_allow_legacy(self):
        d = _tok(["merhaba"]).to_dict()
        d["offset"] = BPETokenizer.OFFSET - 2  # eski OFFSET=6
        with self.assertRaises(ValueError):
            BPETokenizer.from_dict(d, allow_legacy=True)

    def test_specials_mismatch_rejected_even_with_allow_legacy(self):
        d = _tok(["merhaba"]).to_dict()
        d["specials"] = d["specials"][:-1]  # eski 6 ozel token
        with self.assertRaises(ValueError):
            BPETokenizer.from_dict(d, allow_legacy=True)

    def test_version_mismatch_rejected_even_with_allow_legacy(self):
        d = _tok(["merhaba"]).to_dict()
        d["version"] = 2
        with self.assertRaises(ValueError):
            BPETokenizer.from_dict(d, allow_legacy=True)

    def test_matching_tokenizer_round_trips(self):
        tok = _tok(["merhaba dunya"])
        d = tok.to_dict()
        reloaded = BPETokenizer.from_dict(d)  # uyumlu: hata vermemeli
        self.assertEqual(tok.encode("Merhaba dünya"), reloaded.encode("Merhaba dünya"))


if __name__ == "__main__":
    unittest.main()
