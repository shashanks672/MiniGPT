import torch
from pathlib import Path

from tokenizer import encode


# -----------------------------
# 1. Locate training data
# -----------------------------

project_root = Path(__file__).resolve().parent.parent

train_path = project_root / "data" / "train.txt"

text = train_path.read_text(encoding="utf-8")


# -----------------------------
# 2. Encode entire dataset
# -----------------------------

data = encode(text)


# -----------------------------
# 3. Convert to PyTorch tensor
# -----------------------------

data = torch.tensor(data, dtype=torch.long)


# -----------------------------
# 4. Context length
# -----------------------------

block_size = 64 #sequence length
batch_size = 4

# -----------------------------
# 5. Pick random position
# -----------------------------

# start = torch.randint(
#     0,
#     len(data) - block_size,
#     (1,)
# ).item()


# -----------------------------
# 6. Create input and target
# -----------------------------

# x = data[start : start + block_size]

# y = data[start + 1 : start + block_size + 1]

def get_batch():
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )
    x = torch.stack([
        data[start : start + block_size]
        for start in starts 
    ])

    y = torch.stack([
        data[start + 1 : start + block_size + 1]
        for start in starts
    ])

    return x, y



