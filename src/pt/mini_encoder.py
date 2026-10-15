import numpy as np
import torch
from torch import nn


class MiniPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MiniEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, nhead: int = 4, dim_feedforward: int = 64, n_layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = MiniPositionalEncoding(d_model, max_len=128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = self.pos(x)
        x = self.encoder(x)
        return self.head(x)


def make_demo_data(seq_len: int = 8, vocab_size: int = 32, batch_size: int = 4):
    rng = np.random.default_rng(7)
    x = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    y = np.roll(x, -1, axis=1)
    return torch.from_numpy(x), torch.from_numpy(y)


def run_demo(steps: int = 25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniEncoder(vocab_size=32, d_model=32, nhead=4, dim_feedforward=64, n_layers=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x, y = make_demo_data()
    x = x.to(device)
    y = y.to(device)

    for step in range(1, steps + 1):
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == 1:
            print(f"step={step:02d} loss={loss.item():.4f}")

    with torch.no_grad():
        preds = model(x)
        next_tokens = preds.argmax(dim=-1)
    print("sample tokens:", next_tokens[0].tolist())
    print("demo ok")


if __name__ == "__main__":
    run_demo()
