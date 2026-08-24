import torch
from pathlib import Path

from model import MiniGPT
from tokenizer import encode, decode


# ============================================================
# Model configuration
# ============================================================

block_size = 64


# ============================================================
# Project paths
# ============================================================

project_root = Path(__file__).resolve().parent.parent

model_path = project_root / "minigpt.pth"


# ============================================================
# Create model
# ============================================================

model = MiniGPT()


# ============================================================
# Load trained model
# ============================================================

model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu"
    )
)


# ============================================================
# Evaluation mode
# ============================================================

model.eval()

print("Model loaded successfully!")


# ============================================================
# Text generation function
# ============================================================

def generate_text(prompt, max_new_tokens=100):

    # --------------------------------------------------------
    # Text → Token IDs
    # --------------------------------------------------------

    token_ids = encode(prompt)

    # --------------------------------------------------------
    # Token IDs → PyTorch tensor
    # --------------------------------------------------------

    context = torch.tensor(
        [token_ids],
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Generate tokens
    # --------------------------------------------------------

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep latest block_size tokens
            context = context[:, -block_size:]

            # Forward pass
            logits = model(context)

            # Get prediction for final token
            logits = logits[:, -1, :]

            # Logits → probabilities
            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            # Choose most likely token
            next_token = torch.argmax(
                probabilities,
                dim=-1,
                keepdim=True
            )

            # Add token to context
            context = torch.cat(
                (
                    context,
                    next_token
                ),
                dim=1
            )

    # --------------------------------------------------------
    # Token IDs → Text
    # --------------------------------------------------------

    generated_text = decode(
        context[0].tolist()
    )

    return generated_text


# ============================================================
# Generate text
# ============================================================

prompt = "CORIOLANUS:"

generated_text = generate_text(
    prompt,
    max_new_tokens=100
)


# ============================================================
# Display result
# ============================================================

print()
print("Prompt:")
print(prompt)

print()
print("Generated text:")
print(generated_text)