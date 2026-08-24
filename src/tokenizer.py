from pathlib import Path


project_root = Path(__file__).resolve().parent.parent

train_path = project_root / "data" / "train.txt"

text = train_path.read_text(encoding="utf-8")

chars = sorted(set(text))

vocab_size = len(chars)


# Character → Integer
stoi = {ch: i for i, ch in enumerate(chars)}

# Integer → Character
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


# Only run these tests when tokenizer.py
# is executed directly.
if __name__ == "__main__":

    print("Vocabulary size:", vocab_size)
    print("Characters:", chars)

    print("ID of 'a':", stoi["a"])
    print("Character at ID 40:", itos[40])

    sample_text = "Hello"

    encoded = encode(sample_text)
    decoded = decode(encoded)

    print("Original:", sample_text)
    print("Encoded:", encoded)
    print("Decoded:", decoded)