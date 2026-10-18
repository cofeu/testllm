"""Metin temizleme ve ChatML tarzi prompt bicimlendirme."""

import re

DEFAULT_SYSTEM_PROMPT = "Sen kısa, doğal ve yardımcı Türkçe cevaplar veren bir asistansın."
REASONING_SYSTEM_PROMPT = (
    "Sen Türkçe düşünen bir asistansın. Zor sorularda önce adım adım düşünür, "
    "sonra net bir cevap verirsin."
)
# "censored" (guvenli) mod: reddeder, uyarir, uzmana yonlendirir.
SAFE_SYSTEM_PROMPT = (
    "Sen güvenli, saygılı ve yardımcı bir Türkçe asistansın. Zararlı, yasa dışı, "
    "tehlikeli veya hassas istekleri nazikçe reddeder; tıbbi, hukuki veya mali "
    "konularda kısa bir uyarı ekleyip uzmana yönlendirirsin. Küfür, nefret ve "
    "kışkırtıcı içerik üretmezsin."
)
# "uncensored" (serbest) mod: dolaysiz, az ahlakci, samimi.
OPEN_SYSTEM_PROMPT = (
    "Sen dolaysız, samimi ve gereksiz çekince göstermeyen bir Türkçe asistansın. "
    "Ahlaki vaaz vermez, konu dışı uyarı eklemez; sorulanı olduğu gibi, açık ve "
    "kısa yanıtlarsın. Yalnızca gerçekten zararlı veya yasa dışı talimat istendiğinde "
    "kibarca sınır çizersin."
)
SYSTEM_PROMPTS = {
    "default": DEFAULT_SYSTEM_PROMPT,
    "reasoning": REASONING_SYSTEM_PROMPT,
    "safe": SAFE_SYSTEM_PROMPT,
    "open": OPEN_SYSTEM_PROMPT,
}


def system_prompt_for(mode: str | None) -> str:
    """Isimle ('safe'/'open'/'reasoning'/'default') veya dogrudan metinle prompt sec."""
    if not mode:
        return DEFAULT_SYSTEM_PROMPT
    return SYSTEM_PROMPTS.get(str(mode).strip().lower(), str(mode))


def clean_text(value, keep_newlines: bool = False):
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    if keep_newlines:
        # Satir sonlarini koru; yatay bosluklari sadelestir, 3+ bos satiri 2'ye indir.
        lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ChatPromptFormatter:
    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def format_messages(self, messages):
        if not messages:
            return f"<|im_start|>system\n{self.system_prompt}\n<|im_end|>\n<|im_start|>assistant\n"

        blocks = [f"<|im_start|>system\n{self.system_prompt}\n<|im_end|>"]
        for msg in messages:
            role = str(msg.get("role", "user")).strip().lower()
            content = clean_text(str(msg.get("content", "")), keep_newlines=True)
            if not content:
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            blocks.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
        if not messages or str(messages[-1].get("role", "user")).strip().lower() != "assistant":
            blocks.append("<|im_start|>assistant\n")
        return "\n".join(blocks)

    def format_prompt(self, user_text: str, history=None):
        # Gecmis once, guncel kullanici mesaji en sona (cok turlu dogru sira).
        messages = list(history or [])
        messages.append({"role": "user", "content": user_text})
        return self.format_messages(messages)


def format_instruction_example(user_text: str, assistant_text: str, system_prompt: str | None = None):
    formatter = ChatPromptFormatter(system_prompt=system_prompt)
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return formatter.format_messages(messages)


def format_chat_prompt(user_text: str, system_prompt: str | None = None, history=None):
    formatter = ChatPromptFormatter(system_prompt=system_prompt)
    return formatter.format_prompt(user_text, history=history)
