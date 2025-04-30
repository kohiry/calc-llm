import re
import torch
from torch.utils.data import Dataset


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def vectorize(text, vocab):
    vec = torch.zeros(len(vocab))
    for token in tokenize(text):
        idx = vocab.get(token, 0)
        vec[idx] += 1
    return vec


# ====================
# ==== Создаём датасет для модели torch =====


class EmotionDataset(Dataset):
    def __init__(self, data):
        self.data = data  # список (vector, label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_vec, y_label = self.data[idx]
        return x_vec, torch.tensor(y_label)


# создаём первую модель
class EmotionClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_clases):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, num_clases)

    def forward(self, x):
        return self.fc(x)


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
