import requests


def main():
    n = 1000
    data_set = ""
    for i in range(1, n):
        for j in range(1, n):
            data_set += f"\n{i} + {j} = {i + j}"
            data_set += f"\n{i} сумма {j} равно {i + j}"
            data_set += f"\n{i} разность {j} равно {i - j}"
            data_set += f"\n{i} деление {j} равно {i / j}"
            data_set += f"\n{i} умножение {j} равно {i * j}"
            data_set += f"\n{i} * {j} = {i * j}"
            data_set += f"\n{i} - {j} = {i - j}"
            data_set += f"\n{i} / {j} = {i / j}"
    with open("data_set.txt", "w") as f:
        f.write(data_set)


def main2():

    # 1. Скачиваем JSON с Hugging Face
    url = "https://huggingface.co/datasets/kiyo0/Anime-Waifu-Persona-Dataset/resolve/main/wifuwork.json"
    response = requests.get(url)
    response.raise_for_status()  # Проверка на ошибки

    # 2. Парсим JSON
    data = response.json()

    # 3. Извлекаем все responses
    all_responses = []
    for record in data["records"]:
        if "output" in record:
            if type(record["output"]["responses"][0]) is str:
                print(record["output"]["responses"])
                all_responses.append("\n".join(record["output"]["responses"]))
            else:
                print(record["output"]["responses"])
                all_responses.append(
                    "\n".join(
                        map(lambda x: x["response"], record["output"]["responses"])
                    )
                )
    print(len(all_responses))
    # 4. Сохраняем в файл
    with open("data_set2.txt", "w", encoding="utf-8") as f:
        for response in all_responses:
            f.write(response + "\n")

    print("✅ Responses сохранены в data_set2.txt")


if __name__ == "__main__":
    main2()
