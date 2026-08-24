import torch
import torch.nn as nn
import math

from dataset import get_batch
from tokenizer import decode

# ============================================================
# Model configuration
# ============================================================

vocab_size = 65
embedding_dim = 128
block_size = 64

num_heads = 4
head_dim = embedding_dim // num_heads

feed_forward_dim = 4 * embedding_dim


# ============================================================
# Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        # Q, K, V projections
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Output projection
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Current sequence length
        # ----------------------------------------------------

        sequence_length = x.shape[1]

        # ----------------------------------------------------
        # Create Q, K, V
        # ----------------------------------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # ----------------------------------------------------
        # Split embedding dimension into multiple heads
        # ----------------------------------------------------

        Q = Q.view(
            Q.shape[0],
            sequence_length,
            num_heads,
            head_dim
        )

        K = K.view(
            K.shape[0],
            sequence_length,
            num_heads,
            head_dim
        )

        V = V.view(
            V.shape[0],
            sequence_length,
            num_heads,
            head_dim
        )

        # ----------------------------------------------------
        # Move heads before sequence length
        # ----------------------------------------------------

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        scores = Q @ K.transpose(-2, -1)

        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        scores = scores / math.sqrt(head_dim)

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------
        # it creates a lower triangular mask
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        # why ? because gpt should not see the next token
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # ----------------------------------------------------
        # Weighted combination of values
        # ----------------------------------------------------

        head_output = attention_weights @ V

        # ----------------------------------------------------
        # Move heads back
        # ----------------------------------------------------

        head_output = head_output.transpose(1, 2)

        # ----------------------------------------------------
        # Combine heads
        # ----------------------------------------------------

        head_output = head_output.contiguous().view(
            x.shape[0],
            sequence_length,
            embedding_dim
        )

        # ----------------------------------------------------
        # Final projection
        # ----------------------------------------------------

        output = self.output_projection(
            head_output
        )

        return output


# ============================================================
# Feed Forward Network
# ============================================================

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                feed_forward_dim
            ),

            nn.GELU(),

            nn.Linear(
                feed_forward_dim,
                embedding_dim
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention()

        # Feed forward network
        self.feed_forward = FeedForward()

        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(
            embedding_dim
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        attention_output = self.attention(x)

        # ----------------------------------------------------
        # Residual connection + LayerNorm
        # ----------------------------------------------------

        x = self.layer_norm_1(
            x + attention_output
        )

        # ----------------------------------------------------
        # Feed Forward
        # ----------------------------------------------------

        feed_forward_output = self.feed_forward(x)

        # ----------------------------------------------------
        # Residual connection + LayerNorm
        # ----------------------------------------------------

        x = self.layer_norm_2(
            x + feed_forward_output
        )

        return x


# ============================================================
# Transformer
# ============================================================

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList([

            TransformerBlock()

            for _ in range(4)

        ])

    def forward(self, x):

        for block in self.blocks:

            x = block(x)

        return x


# ============================================================
# MiniGPT
# ============================================================

class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Token Embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # ----------------------------------------------------
        # Position Embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        self.transformer = Transformer()

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        # ----------------------------------------------------
        # Language Model Head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Current sequence length
        # ----------------------------------------------------

        sequence_length = x.shape[1]

        # ----------------------------------------------------
        # Token IDs → Token vectors
        # ----------------------------------------------------

        token_vectors = self.token_embedding(x)

        # ----------------------------------------------------
        # Position IDs
        # ----------------------------------------------------

        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        # ----------------------------------------------------
        # Position IDs → Position vectors
        # ----------------------------------------------------

        position_vectors = self.position_embedding(
            positions
        )

        # ----------------------------------------------------
        # Token + Position information
        # ----------------------------------------------------

        x = token_vectors + position_vectors

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        x = self.transformer(x)

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        x = self.final_layer_norm(x)

        # ----------------------------------------------------
        # Embeddings → Vocabulary scores
        # ----------------------------------------------------

        logits = self.lm_head(x)

        return logits


# ============================================================
# Get training batch
# ============================================================

x, y = get_batch()

print("Input shape:", x.shape)


# ============================================================
# Create model
# ============================================================

model = MiniGPT()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)


# ============================================================
# Loss function
# ============================================================

loss_function = nn.CrossEntropyLoss()


# ============================================================
# Training loop
# ============================================================

num_steps = 5000

for step in range(num_steps):

    # --------------------------------------------------------
    # Get a new training batch
    # --------------------------------------------------------

    x, y = get_batch()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(x)

    # --------------------------------------------------------
    # Flatten logits
    # --------------------------------------------------------

    logits = logits.view(
        -1,
        vocab_size
    )

    # --------------------------------------------------------
    # Flatten targets
    # --------------------------------------------------------

    y = y.view(-1)

    # --------------------------------------------------------
    # Calculate loss
    # --------------------------------------------------------

    loss = loss_function(
        logits,
        y
    )

    # --------------------------------------------------------
    # Clear old gradients
    # --------------------------------------------------------

    optimizer.zero_grad()

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Update model weights
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # Print loss
    # --------------------------------------------------------

    if step % 100 == 0:

        print(
            f"Step {step}, Loss: {loss.item():.4f}"
        )

# =========================
# Save trained model
# =========================

torch.save(
    model.state_dict(),
    "minigpt.pth"
)

print("Model saved successfully!")

# ============================================================
# Text Generation
# ============================================================

model.eval()

# ------------------------------------------------------------
# Start with one token
# ------------------------------------------------------------

context = torch.tensor([[0]])

# ------------------------------------------------------------
# Number of tokens to generate
# ------------------------------------------------------------

max_new_tokens = 50


# ------------------------------------------------------------
# Generate tokens
# ------------------------------------------------------------

with torch.no_grad():

    for _ in range(max_new_tokens):

        # ----------------------------------------------------
        # Keep only the latest block_size tokens
        # ----------------------------------------------------

        context = context[:, -block_size:]

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(context)

        # ----------------------------------------------------
        # Get predictions for the last token
        # ----------------------------------------------------

        logits = logits[:, -1, :]

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # ----------------------------------------------------
        # Choose the most likely token
        # ----------------------------------------------------

        next_token = torch.argmax(
            probabilities,
            dim=-1,
            keepdim=True
        )

        # ----------------------------------------------------
        # Add new token to context
        # ----------------------------------------------------

        context = torch.cat(
            (
                context,
                next_token
            ),
            dim=1
        )


# ============================================================
# Print generated tokens
# ============================================================

print("Generated token IDs:")
print(context)

generated_ids = context[0].tolist()
generated_text = decode(generated_ids)

print("Generated text:")
print(generated_text)