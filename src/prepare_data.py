from pathlib import Path


# -----------------------------
# 1. Locate the dataset
# -----------------------------

project_root = Path(__file__).resolve().parent.parent

input_path = project_root / "data" / "input.txt"
train_path = project_root / "data" / "train.txt"
val_path = project_root / "data" / "val.txt"


# -----------------------------
# 2. Read the raw text
# -----------------------------

text = input_path.read_text(encoding="utf-8")

print(f"Total characters: {len(text):,}")


# -----------------------------
# 3. Split into train / validation
# -----------------------------

split_index = int(len(text) * 0.90)

train_text = text[:split_index]
val_text = text[split_index:]


# -----------------------------
# 4. Save the datasets
# -----------------------------

train_path.write_text(train_text, encoding="utf-8")
val_path.write_text(val_text, encoding="utf-8")


print(f"Training characters: {len(train_text):,}")
print(f"Validation characters: {len(val_text):,}")

print("Dataset preparation complete!")