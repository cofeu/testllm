"""Korpustan BPE tokenizer egitir ve JSON olarak kaydeder.

Egitim (train.main) ile ayni veri hattini kullanir: veri dosyalarini okur,
temizler ve `BPETokenizer.from_corpus` ile egitir. Boylece cache'lenen
`data/tokenizer.json` tam korpusla (satir sonlari dahil) tutarli olur.

Kullanim:
    python scripts/train_tokenizer.py --out data/tokenizer.json \
        --data data/instruct.jsonl --data data/reasoning.jsonl --data data/data_large.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.py.normal.corpus import build_assistant_only_corpus, load_jsonl_records
from src.py.normal.prompt import DEFAULT_SYSTEM_PROMPT
from src.py.normal.tokenizer import BPETokenizer


def main(argv=None):
    ap = argparse.ArgumentParser(description="Korpustan BPE tokenizer egit")
    ap.add_argument("--data", action="append", default=None, metavar="PATH")
    ap.add_argument("--out", default=str(ROOT / "data" / "tokenizer.json"))
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--max-examples", type=int, default=100000)
    args = ap.parse_args(argv)

    paths = args.data or [
        str(ROOT / "data" / "instruct.jsonl"),
        str(ROOT / "data" / "reasoning.jsonl"),
        str(ROOT / "data" / "data_large.jsonl"),
    ]
    dialogues = []
    for path in paths:
        rows = load_jsonl_records(path)
        print(f"  {path}: {len(rows)} kayit", flush=True)
        dialogues += rows

    corpus = build_assistant_only_corpus(
        dialogues, system_prompt=DEFAULT_SYSTEM_PROMPT, max_examples=args.max_examples
    )
    texts = [c["user"] for c in corpus] + [c["assistant"] for c in corpus]
    print(f"korpus: {len(corpus)} ornek, {len(texts)} metin", flush=True)

    tokenizer = BPETokenizer.from_corpus(texts, vocab_size=args.vocab)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, ensure_ascii=False)
    tok_len = len(tokenizer.encode("\n\n"))
    print(f"tokenizer egitildi: vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)}"
          f" (\\n\\n -> {tok_len} token) -> {args.out}")
    return tokenizer


if __name__ == "__main__":
    main()
