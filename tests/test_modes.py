import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.py.normal.corpus import build_assistant_only_corpus, load_jsonl_records
from src.py.normal.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    OPEN_SYSTEM_PROMPT,
    SAFE_SYSTEM_PROMPT,
    system_prompt_for,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SystemPromptTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(system_prompt_for("safe"), SAFE_SYSTEM_PROMPT)
        self.assertEqual(system_prompt_for("open"), OPEN_SYSTEM_PROMPT)
        self.assertEqual(system_prompt_for(None), DEFAULT_SYSTEM_PROMPT)
        # Bilinmeyen deger dogrudan prompt olarak kullanilir.
        self.assertEqual(system_prompt_for("Özel prompt"), "Özel prompt")


class RecordLoaderTests(unittest.TestCase):
    def test_records_preserve_system(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"system": "SYS-A", "turns": [["soru", "cevap burada"]]}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"turns": [["soru2", "cevap2 burada"]]}, ensure_ascii=False) + "\n")
            recs = load_jsonl_records(path)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["system"], "SYS-A")
        self.assertIsNone(recs[1]["system"])

    def test_instruction_corpus_per_dialogue_system(self):
        recs = [
            {"system": SAFE_SYSTEM_PROMPT, "turns": [["Aynı soru", "Güvenli cevap burada"]]},
            {"system": OPEN_SYSTEM_PROMPT, "turns": [["Aynı soru", "Açık cevap burada"]]},
        ]
        corpus = build_assistant_only_corpus(recs)
        # Ayni soru iki farkli modda ayri ayri korunur.
        self.assertEqual(len(corpus), 2)
        systems = {c["system"] for c in corpus}
        self.assertEqual(systems, {SAFE_SYSTEM_PROMPT, OPEN_SYSTEM_PROMPT})

    def test_default_system_used_when_missing(self):
        recs = [{"system": None, "turns": [["soru", "cevap burada"]]}]
        corpus = build_assistant_only_corpus(recs, system_prompt=DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(corpus[0]["system"], DEFAULT_SYSTEM_PROMPT)


class ModeDataTests(unittest.TestCase):
    def test_generator_pairs(self):
        gen = _load(ROOT / "scripts" / "make_mode_data.py")
        rows = gen.generate()
        self.assertGreater(len(rows), 0)
        self.assertEqual(len(rows) % 2, 0)
        systems = {r["system"] for r in rows}
        self.assertEqual(systems, {SAFE_SYSTEM_PROMPT, OPEN_SYSTEM_PROMPT})
        # Ayni soru, iki farkli cevap.
        by_q = {}
        for r in rows:
            q = r["turns"][0][0]
            by_q.setdefault(q, set()).add(r["turns"][0][1])
        self.assertTrue(any(len(v) == 2 for v in by_q.values()))


if __name__ == "__main__":
    unittest.main()
