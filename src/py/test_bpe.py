from tokenizer import BPETokenizer


def test_bpe_roundtrip():
    tok = BPETokenizer.from_corpus(["Merhaba dünya", "Merhaba! Nasıl gidiyor?"], vocab_size=128)
    text = "Merhaba dünya"
    ids = tok.encode(text)
    assert ids
    decoded = tok.decode(ids)
    assert decoded.strip() == text.strip()
    assert tok.vocab_size >= 128
    assert tok.BOS == 1 and tok.EOS == 2


if __name__ == "__main__":
    test_bpe_roundtrip()
    print("bpe tokenizer ok")
