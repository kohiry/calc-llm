from third.emot_data_set import (
    dataset,
)
import pickle
from third.utils import EmotionClassifier, EmotionDataset, evaluate, tokenize, vectorize
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


# ===== заготовка =====

all_tokens = []

for item in dataset:
    tokens = tokenize(item["text"])
    all_tokens.extend(tokens)


word_counts = Counter(all_tokens)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common())}
vocab["<UNK>"] = 0

print(f"Слов в словаре: {len(vocab)}")


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


input_dim = len(vocab)  # РАзмерность входного вектор
num_classes = len(label2idx)  # 4 эмоции

model = EmotionClassifier(input_dim, num_classes)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# accuracy


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
torch.save(model.state_dict(), "model/emotion_model.pth")
print("✅ Модель сохранена в файл emotion_model.pth")


with open("model/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("model/labels.pkl", "wb") as f:
    pickle.dump(label2idx, f)
