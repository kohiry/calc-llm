import torch

import torch.nn as nn
from torch.nn import functional as F

BLOCK_SIZE = 128
BATCH_SIZE = 32
MAX_ITERS = 50_000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
EVAL_ITERS = 200
torch.manual_seed(1337)
N_EMBED = 96
DROPOUT = 0.2
N_HEAD = 6
N_LAYER = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


class Head(nn.Module):
    """one head self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C**-0.5)  # (B, T, C) @ (B,C, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class FeedForward(nn.Module):
    """a simple liner layer followed by non-linearty"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multiple head of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    """Transformer block: communiction followed by computation"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(2 * BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(
            *[Block(N_EMBED, n_head=N_HEAD) for _ in range(N_LAYER)],
        )
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (Batch, Time,Channel)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device) % (BLOCK_SIZE * 2)
        )
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)  # get the prediction
            # focus only on the last time stamp
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def get_batch(
    split,
    train_data,
    val_data,
):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [data[i : i + BLOCK_SIZE] for i in ix],
    )
    y = torch.stack(
        [data[i + 1 : i + BLOCK_SIZE + 1] for i in ix],
    )
    x, y = x.to(device), y.to(device)
    return x, y


def encode(stoi, s):
    return [stoi[c] for c in s]


def decode(itos, l):
    return "".join([itos[i] for i in l])


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def main():
    with open("data_set2.txt", "r") as f:
        ds = f.read()
    chars = sorted(list(set(ds)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor(encode(stoi, ds), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

    # torch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

    # train
    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(m, train_data, val_data)
            print(
                f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses['val']:.4f}"
            )
        xb, yb = get_batch("train", train_data, val_data)
        print(f"step {iter}, max {MAX_ITERS} ")

        # evalute the loss
        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(itos, m.generate(idx, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
