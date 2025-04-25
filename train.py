import torch

import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (Batch, Time,Channel)
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
            logits, _ = self(idx)  # get the prediction
            # focus only on the last time stamp
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


BLOCK_SIZE = 8
BATCH_SIZE = 32
MAX_ITERS = 300000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
EVAL_ITERS = 200
torch.manual_seed(1337)


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

    m = BigramLanguageModel(vocab_size)

    # torch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    # train
    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(m, train_data, val_data)
            print(
                f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses['val']:.4f}"
            )
        xb, yb = get_batch("train", train_data, val_data)

        # evalute the loss
        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    idx = torch.zeros((1, 1), dtype=torch.long)
    print(decode(itos, m.generate(idx, max_new_tokens=50)[0].tolist()))


if __name__ == "__main__":
    main()
