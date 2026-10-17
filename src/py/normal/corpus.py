"""Veri okuma, temizleme ve instruction korpusu olusturma."""

import json
import os

import numpy as np
import torch

from .config import TrainConfig
from .prompt import DEFAULT_SYSTEM_PROMPT, clean_text
from .tokenizer import BPETokenizer


def _has_special_token(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in BPETokenizer.SPECIAL_STRINGS)


def messages_to_turns(messages):
    """ChatML tarzi `messages` listesini [ [user, assistant], ... ] turlarina cevirir.

    Sistem mesajlari atlanir; ard arda gelen kullanici mesajlari birlestirilir;
    sonu asistanla eslesmeyen kullanici mesaji yok sayilir.
    """
    turns = []
    pending = None
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = clean_text(msg.get("content", ""), keep_newlines=True)
        if not content:
            continue
        if role == "user":
            pending = content if pending is None else f"{pending}\n{content}"
        elif role == "assistant" and pending is not None:
            turns.append([pending, content])
            pending = None
    return turns


def load_jsonl_dialogues(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if "turns" in payload:
                rows.append(payload["turns"])
            elif "messages" in payload:
                turns = messages_to_turns(payload["messages"])
                if turns:
                    rows.append(turns)
    return rows


def build_instruction_corpus(dialogues, system_prompt: str | None = None, max_turns: int | None = None, max_history: int = 8):
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    corpus = []
    seen = set()

    for dialogue in dialogues[:max_turns] if max_turns else dialogues:
        if not isinstance(dialogue, (list, tuple)):
            continue
        history = []  # onceki turlar: [{"role","content"}, ...]
        for turn in dialogue:
            if not isinstance(turn, (list, tuple)) or len(turn) != 2:
                continue
            user, assistant = turn
            user_text = clean_text(user, keep_newlines=True)
            assistant_text = clean_text(assistant, keep_newlines=True)
            if len(user_text) < 3 or len(assistant_text) < 5:
                continue
            if _has_special_token(user_text) or _has_special_token(assistant_text):
                continue
            key = (tuple((m["role"], m["content"]) for m in history),
                   user_text.lower(), assistant_text.lower())
            if key not in seen:
                seen.add(key)
                corpus.append({
                    "system": system_prompt,
                    "user": user_text,
                    "assistant": assistant_text,
                    "history": list(history),
                })
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant_text})
            if max_history and len(history) > max_history:
                history = history[-max_history:]
    return corpus


def build_assistant_only_corpus(dialogues, system_prompt: str | None = None, max_examples: int | None = None):
    examples = build_instruction_corpus(dialogues, system_prompt=system_prompt)

    clean = []
    seen = set()
    for item in examples:
        user = clean_text(item["user"], keep_newlines=True)
        assistant = clean_text(item["assistant"], keep_newlines=True)
        if len(user) < 4 or len(assistant) < 4:
            continue
        if assistant.lower().startswith(user.lower()[:20]):
            continue
        if _has_special_token(user) or _has_special_token(assistant):
            continue
        history = [
            {"role": m.get("role"), "content": clean_text(m.get("content", ""), keep_newlines=True)}
            for m in item.get("history", [])
            if clean_text(m.get("content", ""), keep_newlines=True)
        ]
        key = (tuple((m["role"], m["content"]) for m in history), user.lower(), assistant.lower())
        if key in seen:
            continue
        seen.add(key)
        clean.append({
            "system": clean_text(item.get("system", system_prompt or ""), keep_newlines=True),
            "user": user,
            "assistant": assistant,
            "history": history,
        })
        # Sinir temizleme (dedup/eleme) sonrasi uygulanir; boylece istenen sayida
        # gecerli ornek elde edilir.
        if max_examples is not None and len(clean) >= max_examples:
            break
    return clean


def build_long_clean_chat_corpus(dialogues, system_prompt: str | None = None, max_examples: int | None = None):
    return build_assistant_only_corpus(dialogues, system_prompt=system_prompt, max_examples=max_examples)


def build_chatml_dataset_file(data_path: str, out_path: str, system_prompt: str | None = None, max_examples: int | None = None):
    dialogues = load_jsonl_dialogues(data_path)
    corpus = build_long_clean_chat_corpus(dialogues, system_prompt=system_prompt, max_examples=max_examples)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in corpus:
            msg = [
                {"role": "system", "content": clean_text(item.get("system", system_prompt or ""), keep_newlines=True)},
                {"role": "user", "content": clean_text(item["user"], keep_newlines=True)},
                {"role": "assistant", "content": clean_text(item["assistant"], keep_newlines=True)},
            ]
            f.write(json.dumps({"messages": msg}, ensure_ascii=False) + "\n")
    return corpus


def load_chatml_dataset(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            messages = payload.get("messages", [])
            if not isinstance(messages, list):
                continue
            rows.append(messages)
    return rows


def make_synthetic_data(cfg: TrainConfig):
    """Encoder-decoder icin (src, tgt_in, tgt_out) sentetik yigin uretir.

    Gercek veri yoksa egitim dongusunun calismasi icin: kaynak ve hedef rastgele
    token dizileri. tgt_in bir kaydirilmis hedeftir (teacher forcing).
    """
    rng = np.random.default_rng(cfg.seed)
    low = BPETokenizer.OFFSET
    src = rng.integers(low, cfg.vocab_size, size=(cfg.batch_size, cfg.block_size), dtype=np.int64)
    tgt_in = rng.integers(low, cfg.vocab_size, size=(cfg.batch_size, cfg.block_size), dtype=np.int64)
    tgt_out = np.roll(tgt_in, -1, axis=1)
    return torch.from_numpy(src), torch.from_numpy(tgt_in), torch.from_numpy(tgt_out)
