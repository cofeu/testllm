"""Metin uretimi: prompt'u token'lar, uretim yapar, ozel token'lari suzup cozer."""

import torch

from .config import GenerationConfig
from .model import DecoderOnlyLM
from .tokenizer import BPETokenizer


def generate_text(model: DecoderOnlyLM, tokenizer: BPETokenizer, prompt: str, generation: GenerationConfig | None = None, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generation = generation or GenerationConfig()
    prompt_text = str(prompt or "")
    if not prompt_text.strip():
        return ""
    if not prompt_text.endswith("\n"):
        prompt_text = prompt_text + "\n"

    prompt_ids = tokenizer.encode(prompt_text)
    if not prompt_ids:
        return ""

    if len(prompt_ids) > model.cfg.block_size:
        prompt_ids = prompt_ids[-model.cfg.block_size:]

    prompt_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    generated = model.generate(
        prompt_tensor,
        max_new_tokens=generation.max_new_tokens,
        temperature=generation.temperature,
        top_k=generation.top_k,
        top_p=generation.top_p,
        min_p=generation.min_p,
        repetition_penalty=generation.repetition_penalty,
        device=device,
        eos_id=tokenizer.EOS,
        forbidden_ids={tokenizer.PAD, tokenizer.BOS, tokenizer.UNK, tokenizer.IM_START},
        stop_ids={tokenizer.IM_END},
        sampler=generation.sampler,
        typical_p=generation.typical_p,
        mirostat_tau=generation.mirostat_tau,
        mirostat_eta=generation.mirostat_eta,
    )
    output_ids = generated[0].tolist()
    continuation_ids = output_ids[len(prompt_ids):]
    visible_ids = [idx for idx in continuation_ids if idx >= tokenizer.OFFSET]
    if not visible_ids:
        return ""

    decoded = tokenizer.decode(visible_ids)
    return decoded.replace("\x00", "").strip()
