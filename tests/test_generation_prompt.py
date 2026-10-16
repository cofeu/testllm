import unittest

from src.pt.decoder_model import (
    BPETokenizer,
    DecoderOnlyLM,
    GenerationConfig,
    TrainConfig,
    build_assistant_only_corpus,
    generate_text,
)


class GenerationPromptTests(unittest.TestCase):
    def test_generate_text_does_not_repeat_prompt_in_output(self):
        tokenizer = BPETokenizer.from_corpus([
            "Merhaba, bugün nasıl yardımcı olabilirim?",
            "Bana kısa bir açıklama ver.",
        ], vocab_size=512)

        cfg = TrainConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_layers=1,
            n_head=4,
            ff_dim=64,
            block_size=32,
            dropout=0.0,
            learning_rate=1e-3,
            batch_size=1,
            epochs=1,
            seed=7,
        )
        model = DecoderOnlyLM(cfg)
        prompt = "Merhaba, bugün nasıl yardımcı olabilirim?"
        out = generate_text(
            model,
            tokenizer,
            prompt,
            generation=GenerationConfig(max_new_tokens=16, temperature=1.0, top_k=10, top_p=0.9, repetition_penalty=1.0, min_p=0.0),
            device="cpu",
        )

        self.assertNotIn(prompt, out)
        self.assertFalse(out.startswith(prompt))

    def test_generate_text_handles_prompts_longer_than_block_size(self):
        tokenizer = BPETokenizer.from_corpus([
            "Merhaba, bugün nasıl yardımcı olabilirim? Lütfen kısa ve net cevap ver.",
            "Python nedir? Kısa ve net açıklama yap.",
        ], vocab_size=512)

        cfg = TrainConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_layers=1,
            n_head=4,
            ff_dim=64,
            block_size=16,
            dropout=0.0,
            learning_rate=1e-3,
            batch_size=1,
            epochs=1,
            seed=7,
        )
        model = DecoderOnlyLM(cfg)
        prompt = "Merhaba, bugün nasıl yardımcı olabilirim? Lütfen kısa ve net cevap ver."
        out = generate_text(
            model,
            tokenizer,
            prompt,
            generation=GenerationConfig(max_new_tokens=16, temperature=1.0, top_k=10, top_p=0.9, repetition_penalty=1.0, min_p=0.0),
            device="cpu",
        )

        self.assertIsInstance(out, str)
        self.assertNotIn(prompt, out)

    def test_build_assistant_only_corpus_rejects_prompt_echoes(self):
        dialogues = [[
            ["Merhaba", "Merhaba, nasılsın?"],
            ["Merhaba", "Merhaba! Ben sana nasıl yardımcı olabilirim?"],
            ["Python nedir?", "Python, programlama dili olarak kullanılır."],
        ]]
        corpus = build_assistant_only_corpus(dialogues, system_prompt="Sen yardımcı asistanısın.")

        # Iki "Merhaba" cevabi prompt echo'su olarak elenir; yalniz Python ornegi kalir.
        self.assertEqual(len(corpus), 1)
        self.assertTrue(all(item["assistant"] for item in corpus))
        self.assertFalse(corpus[0]["assistant"].lower().startswith("merhaba"))
        self.assertTrue(corpus[0]["assistant"].lower().startswith("python"))


if __name__ == "__main__":
    unittest.main()
