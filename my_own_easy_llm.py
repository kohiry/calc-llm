from datasets import load_dataset
from collections import Counter
import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn

dataset = load_dataset("daily_dialog")

# ==== vars =====

BATCH_SIZE = 16
EMBEDDING_DIM = 32
HIDDEN_SIZE = 64
LR = 0.001

# ===============


def get_pairs():
    pairs = []

    for dialog in dataset["train"]:
        utterances = dialog["dialog"]
        for i in range(len(utterances) - 1):
            input_text = utterances[i]
            target_text = utterances[i + 1]
            pairs.append((input_text, target_text))
    return pairs


pairs = get_pairs()


def vocabular(pairs: list[tuple[str, str]]):
    counter = Counter()

    for input_text, target_text in pairs:
        counter.update(input_text.split())
        counter.update(target_text.split())
    min_freq = 5
    words = [word for word, freq in counter.items() if freq >= min_freq]
    # Спец токены
    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

    vocab = {word: idx for idx, word in enumerate(special_tokens + words)}
    print(f"Размер словаря: {len(vocab)}")

    return vocab


vocab = vocabular(pairs)

# tokenizer


def tokenize(text, vocab):
    tokens = []
    for word in text.split():
        token = vocab.get(word, vocab["<unk>"])  # если слова нет в словаре
        tokens.append(token)
    return tokens


def tokenize_pairs(pairs):
    res = []
    for input_text, target_text in pairs:
        input_tokens = tokenize(input_text, vocab)
        target_tokens = (
            [vocab["<sos>"]] + tokenize(target_text, vocab) + [vocab["<eos>"]]
        )
        res.append((input_tokens, target_tokens))
    return res


tokenized_pairs = tokenize_pairs(pairs)

# ==== DATASET CLASS FOR TORCH AND PADDING =====


class DialogueDataset(Dataset):
    def __init__(self, tokenized_pairs: list[list[int]]):
        self.tokenized_pairs = tokenized_pairs

    def __len__(self):
        return len(self.tokenized_pairs)

    def __getitem__(self, idx):
        src, tgt = self.tokenized_pairs[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


dataset = DialogueDataset(tokenized_pairs)


# PADDING
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    max_len = max(
        max(len(seq) for seq in src_batch), max(len(seq) for seq in tgt_batch)
    )

    # Паддим src и tgt до одной общей длины
    src_batch = [
        torch.cat(
            [seq, torch.full((max_len - len(seq),), vocab["<pad>"], dtype=torch.long)]
        )
        for seq in src_batch
    ]
    tgt_batch = [
        torch.cat(
            [seq, torch.full((max_len - len(seq),), vocab["<pad>"], dtype=torch.long)]
        )
        for seq in tgt_batch
    ]

    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    return {
        "src": src_batch,
        "tgt": tgt_batch,
    }


train_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# Simple RNN model imp


class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleRNNModel, self).__init__()

        # Embed
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)

        # RNN
        self.rnn = nn.GRU(EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)

        # Линейный вход
        self.fc = nn.Linear(HIDDEN_SIZE, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src)  # (batch_size, seq_len, embedding_dim
        out, _ = self.rnn(embedded)  # outputs: (batch_size, seq_len, hidden_size)
        logits = self.fc(out)  # logits: (batch_size, seq_len, vocab_size)
        # logits - предсказание на каждый токен
        return logits


model = SimpleRNNModel(len(vocab))


criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# training

for epoch in range(1):
    total_loss = 0
    for i, batch in enumerate(train_loader):
        print(f"Осталось до конца эпохи № {epoch}: {i} из {len(train_loader)}")
        src_batch = batch["src"]
        tgt_batch = batch["tgt"]

        optimizer.zero_grad()

        logits = model(src_batch)
        logits = logits[:, :-1, :].contiguous().view(-1, len(vocab))
        targets = tgt_batch[:, 1:].contiguous().view(-1)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
    checkpoint_data = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "vocab": vocab,
    }
    torch.save(checkpoint_data, f"checkpoint_epoch_{epoch+1}.pth")
    print(f"Сохранён чекпоинт после эпохи {epoch+1}")


save_data = {
    "model_state": model.state_dict(),  # веса модели
    "vocab": vocab,  # твой словарь
}

torch.save(save_data, "checkpoint.pth")
print("Модель и словарь сохранены!")
