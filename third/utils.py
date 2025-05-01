import re
import torch
from torch.utils.data import Dataset


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def vectorize(text, vocab, max_len=20):
    tokens = tokenize(text)
    indices = [vocab.get(token, 0) for token in tokens]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor(indices)


# ====================
# ==== Создаём датасет для модели torch =====


class EmotionDataset(Dataset):
    def __init__(self, data):
        self.data = data  # список (vector, label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_vec, y_label = self.data[idx]
        return x_vec.long(), torch.tensor(y_label).long()


# создаём первую модель
class EmotionClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        last_hidden = hidden[-1]
        out = self.fc(last_hidden)
        return out


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            logits = model(x_batch)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
