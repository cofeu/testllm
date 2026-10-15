from src.pt.decoder_model import BPETokenizer


def test_bpe_roundtrip():
    tok = BPETokenizer.from_corpus([
        "Merhaba dünya",
        "PyTorch ile BPE tokenizer çalışıyor.",
    ], vocab_size=256)
    text = "Merhaba dünya"
    ids = tok.encode(text)
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert decoded.strip() == text.strip()
    assert tok.vocab_size >= 256


if __name__ == "__main__":
    test_bpe_roundtrip()
    print("pt bpe ok")
