import json

PROMPT_PREFIX = "K: "
RESPONSE_PREFIX = "A: "


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def normalize_turns(turns):
    out = []
    for pair in turns or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        user = normalize_text(pair[0])
        assistant = normalize_text(pair[1])
        if not user or not assistant:
            continue
        out.append((user, assistant))
    return out


def load_dialogues(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            turns = obj.get("turns", [])
            cleaned = normalize_turns(turns)
            if cleaned:
                recs.append(cleaned)
    return recs


def build_stream(tokenizer, turns):
    """Bir kaydi tek bir token akisina cevirir:
    <bos> K: soru\n A: cevap <eos> K: soru ..."""
    ids = [tokenizer.BOS]
    for user, assistant in normalize_turns(turns):
        text = PROMPT_PREFIX + user + "\n" + RESPONSE_PREFIX + assistant
        ids.extend(tokenizer.encode(text))
        ids.append(tokenizer.EOS)
    return ids


def make_windows(tokens, block_size, stride=None):
    """Akisi (girdi, hedef) pencere pareja boler; hedefler 1 token kaydirilmistir."""
    stride = stride or block_size
    out = []
    i = 0
    n = len(tokens)
    while i < n - 1:
        end = min(i + block_size, n)
        seg = tokens[i:end]
        if len(seg) >= 2:
            out.append((seg[:-1], seg[1:]))
        if end >= n:
            break
        i += stride
    return out


def build_dataset(tokenizer, dialogues, block_size):
    windows = []
    for turns in dialogues:
        stream = build_stream(tokenizer, turns)
        windows.extend(make_windows(stream, block_size))
    return windows


def split_train_val(windows, val_frac, seed):
    import random

    rng = random.Random(seed)
    idx = list(range(len(windows)))
    rng.shuffle(idx)
    n_val = max(1, int(len(windows) * val_frac)) if len(windows) > 1 else 0
    val_idx = set(idx[:n_val])
    train = [windows[i] for i in idx if i not in val_idx]
    val = [windows[i] for i in idx if i in val_idx]
    return train, val
