def encode(stoi, s):
    return [stoi[c] for c in s]


def decode(itos, l):
    return "".join([itos[i] for i in l])


def main():
    with open("data_set.txt", "r") as f:
        ds = f.read()
    chars = sorted(list(set(ds)))
    vocab_size = len(chars)
    print("".join(chars))
    print(vocab_size)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    print(encode(stoi, "123 + 123"))
    print(decode(itos, encode(stoi, "123 + 123")))


if __name__ == "__main__":
    main()
