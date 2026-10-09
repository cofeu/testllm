import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "py"))

from config import Config
from train import apply_args


class ApplyArgsResumeTests(unittest.TestCase):
    def test_resume_keeps_runtime_overrides(self):
        cfg = Config()
        args = argparse.Namespace(
            steps=123,
            log_every=9,
            eval_every=17,
            save_every=19,
            data="custom/data.jsonl",
            ckpt="custom/checkpoint.pkl",
            batch=7,
            lr=0.002,
            block=None,
            d_model=None,
            layers=None,
            heads=None,
            ffn=None,
            seed=99,
        )

        cfg = apply_args(cfg, args, arch=False)

        self.assertEqual(cfg.max_steps, 123)
        self.assertEqual(cfg.log_every, 9)
        self.assertEqual(cfg.eval_every, 17)
        self.assertEqual(cfg.save_every, 19)
        self.assertEqual(cfg.data_path, "custom/data.jsonl")
        self.assertEqual(cfg.ckpt_path, "custom/checkpoint.pkl")
        self.assertEqual(cfg.batch_size, 7)
        self.assertEqual(cfg.lr, 0.002)
        self.assertEqual(cfg.seed, 99)


if __name__ == "__main__":
    unittest.main()
