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


if __name__ == "__main__":
    main()
