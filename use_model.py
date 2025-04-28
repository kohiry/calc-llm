from my_own_easy_llm import tokenize, SimpleRNNModel
import torch


def generate_response(model, vocab, input_text, max_len=20, device="cpu"):
    model.eval()  # Переводим модель в режим оценки

    # Обратный словарь: индекс -> токен
    inv_vocab = {idx: token for token, idx in vocab.items()}

    # Токенизируем входной текст
    input_tokens = tokenize(input_text, vocab)
    input_tensor = (
        torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
    )  # (1, seq_len)

    # Прогоняем через эмбеддинг и GRU
    with torch.no_grad():
        embedded = model.embedding(input_tensor)
        output, hidden = model.rnn(embedded)

        # Теперь будем генерировать токены один за другим
        generated_tokens = [vocab["<sos>"]]
        for _ in range(max_len):
            prev_token = torch.tensor([[generated_tokens[-1]]], dtype=torch.long).to(
                device
            )
            prev_embedded = model.embedding(prev_token)
            out, hidden = model.rnn(prev_embedded, hidden)
            logits = model.fc(out.squeeze(1))
            next_token = logits.argmax(dim=-1).item()

            # if next_token == vocab["<eos>"]:
            #    break

            generated_tokens.append(next_token)

    # Декодируем токены в текст
    response_tokens = [
        inv_vocab.get(token, "<unk>") for token in generated_tokens[1:]
    ]  # без <sos>
    response_text = generated_tokens
    return response_text


# Загрузка модели и словаря
checkpoint = torch.load("checkpoint_epoch_4.pth", map_location="cpu")
vocab = checkpoint["vocab"]
model = SimpleRNNModel(len(checkpoint["vocab"]))
model.load_state_dict(checkpoint["model_state"])

# Генерация ответа
input_text = "Hello, Adam"
response = generate_response(model, vocab, input_text)
print("Response:", response)
