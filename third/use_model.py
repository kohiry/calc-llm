import pickle
import torch

from third.utils import EmotionClassifier

# --- Загрузка словарей ---
with open("model/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("model/labels.pkl", "rb") as f:
    label2idx = pickle.load(f)
idx2label = {v: k for k, v in label2idx.items()}

# --- Параметры модели ---
input_dim = len(vocab)
output_dim = len(label2idx)


# --- Загрузка модели ---
model = EmotionClassifier(input_dim, output_dim)
model.load_state_dict(torch.load("model/emotion_model.pth"))
model.eval()


def prediction_emotion(text, model, vocab):
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

    return idx2label[prediction]


while True:
    text = input("Введите фразу (или 'выход'): ")
    if text.lower() == "выход":
        break

    emotion = prediction_emotion(text, model, vocab)
    print(f"🤖 Эмоция: {emotion}\n")
