from third.emot_data_set import (
    dataset,
)
from collections import Counter
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


# ===== заготовка =====
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


all_tokens = []

for item in dataset:
    tokens = tokenize(item["text"])
    all_tokens.extend(tokens)


word_counts = Counter(all_tokens)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common())}
vocab["<UNK>"] = 0

print(f"Слов в словаре: {len(vocab)}")


def vectorize(text, vocab):
    vec = torch.zeros(len(vocab))
    for token in tokenize(text):
        idx = vocab.get(token, 0)
        vec[idx] += 1
    return vec


label2idx = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "surprise": 3,
}

data = []

for item in dataset:
    x_vec = vectorize(item["text"], vocab)
    y_label = label2idx[item["label"]]
    data.append((x_vec, y_label))

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


# 80% на обучение, 20 на тест
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = random_split(data, [train_size, test_size])

# обопрачиваем в Dataset
train_dataset = EmotionDataset(train_data)
test_dataset = EmotionDataset(test_data)

# Создаём DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=8)


# создаём первую модель
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_clases):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_clases)

    def forward(self, x):
        return self.fc(x)


input_dim = len(vocab)  # РАзмерность входного вектор
num_classes = len(label2idx)  # 4 эмоции

model = EmotionClassifier(input_dim, num_classes)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# accuracy


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


# алгоритм тренеровки
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {avg_loss:.4f}")

evaluate(model, test_loader)


def prediction_emotion(text, model, vocab, label2idx):
    model.eval()

    words = text.lower().split()
    vector = torch.zeros(len(vocab))

    for word in words:
        if word in vocab:
            idx = vocab[word]
            vector[idx] += 1

    with torch.no_grad():
        logits = model(vector.unsqueeze(0))
        prediction = torch.argmax(logits, dim=1).item()

    idx2label = {v: k for k, v in label2idx.items()}
    print(idx2label)
    return idx2label[prediction]


while True:
    text = input("Введите фразу (или 'выход'): ")
    if text.lower() == "выход":
        break

    emotion = prediction_emotion(text, model, vocab, label2idx)
    print(f"🤖 Эмоция: {emotion}\n")
