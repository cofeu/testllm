import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    path = ROOT / "scripts" / "make_reasoning_data.py"
    spec = importlib.util.spec_from_file_location("make_reasoning_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReasoningDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = _load_generator()

    def test_generates_multiline_step_answers(self):
        rows = self.gen.generate(per_category=5, seed=7)
        self.assertGreater(len(rows), 0)
        multiline = [r for r in rows if "\n" in r["turns"][0][1]]
        # Orneklerin buyuk cogunlugu adim adim (cok satirli) olmali.
        self.assertGreater(len(multiline), len(rows) // 2)
        self.assertTrue(any("Sonuç" in r["turns"][0][1] for r in rows))

    def test_deterministic_for_same_seed(self):
        a = self.gen.generate(per_category=5, seed=42)
        b = self.gen.generate(per_category=5, seed=42)
        self.assertEqual(a, b)

    def test_turns_shape(self):
        rows = self.gen.generate(per_category=3, seed=1)
        for row in rows:
            self.assertIn("turns", row)
            turn = row["turns"][0]
            self.assertEqual(len(turn), 2)
            self.assertTrue(turn[0].strip())
            self.assertTrue(turn[1].strip())


if __name__ == "__main__":
    unittest.main()
