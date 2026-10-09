import json
from pathlib import Path

from dataset import build_dataset, load_dialogues
from generate import format_prompt, sample_next


def _write_sample(path: Path):
    rows = [
        {"turns": [["Merhaba", "Merhaba! Nasıl yardımcı olabilirim?"], ["Bana Türkçe kısa cevap ver.", "Tabii, kısa ve net cevap veririm."]]},
        {"turns": [["Selam", "Selam!" ]]},
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_data_loading_and_prompt_format():
    sample = Path("/tmp/testllm_dataset.jsonl")
    _write_sample(sample)
    dialogues = load_dialogues(str(sample))
    assert len(dialogues) == 2
    assert all(len(turns) >= 1 for turns in dialogues)
    prompt = format_prompt("Merhaba")
    assert prompt.startswith("K: Merhaba") and "\nA:" in prompt

    toks = build_dataset
    assert callable(toks)


def test_sampling_keeps_valid_token_choice():
    logits = [0.2, 0.7, 0.1]
    choice = sample_next(logits, temperature=0.7, top_k=2, rng=None)
    assert 0 <= choice < len(logits)


if __name__ == "__main__":
    test_data_loading_and_prompt_format()
    test_sampling_keeps_valid_token_choice()
    print("improvement checks passed")
