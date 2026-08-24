import torch
import torch.nn as nn
import math

from dataset import get_batch


# =========================
# Configuration
# =========================

vocab_size = 65
embedding_dim = 128
block_size = 64
num_heads = 4

head_dim = embedding_dim // num_heads


# =========================
# Multi-Head Attention
# =========================

class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        # Q, K, V projections
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        # Final output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, x):

        # -------------------------
        # 1. Create Q, K, V
        # -------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        print("Q shape:", Q.shape)
        print("K shape:", K.shape)
        print("V shape:", V.shape)


        # -------------------------
        # 2. Split into heads
        # -------------------------

        Q = Q.view(
            4,
            block_size,
            num_heads,
            head_dim
        )

        K = K.view(
            4,
            block_size,
            num_heads,
            head_dim
        )

        V = V.view(
            4,
            block_size,
            num_heads,
            head_dim
        )


        # -------------------------
        # 3. Move heads before sequence
        # -------------------------

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        print("Multi-head Q shape:", Q.shape)
        print("Multi-head K shape:", K.shape)
        print("Multi-head V shape:", V.shape)


        # -------------------------
        # 4. Attention scores
        # -------------------------

        scores = Q @ K.transpose(-2, -1)

        print("Scores shape:", scores.shape)


        # -------------------------
        # 5. Scale scores
        # -------------------------

        scores = scores / math.sqrt(head_dim)

        print("Scaled scores shape:", scores.shape)


        # -------------------------
        # 6. Causal mask
        # -------------------------

        mask = torch.tril(
            torch.ones(block_size, block_size)
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        print("Masked scores shape:", scores.shape)


        # -------------------------
        # 7. Softmax
        # -------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        print(
            "Attention weights shape:",
            attention_weights.shape
        )


        # -------------------------
        # 8. Weighted combination
        # -------------------------

        head_output = attention_weights @ V

        print(
            "Head output shape:",
            head_output.shape
        )


        # -------------------------
        # 9. Combine heads
        # -------------------------

        head_output = head_output.transpose(1, 2)

        head_output = head_output.contiguous().view(
            4,
            block_size,
            embedding_dim
        )

        print(
            "Combined heads shape:",
            head_output.shape
        )


        # -------------------------
        # 10. Output projection
        # -------------------------

        output = self.output_projection(head_output)

        print(
            "Multi-head attention output shape:",
            output.shape
        )

        return output


# =========================
# Token Embedding
# =========================

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)


# =========================
# Position Embedding
# =========================

position_embedding = nn.Embedding(
    num_embeddings=block_size,
    embedding_dim=embedding_dim
)


# =========================
# Get batch
# =========================

x, y = get_batch()

print("X shape:", x.shape)


# =========================
# Token embedding
# =========================

token_embedding = embedding(x)

print(
    "Token embedding shape:",
    token_embedding.shape
)


# =========================
# Position embedding
# =========================

positions = torch.arange(block_size)

position_vectors = position_embedding(positions)

print(
    "Position vectors shape:",
    position_vectors.shape
)


# =========================
# Final embedding
# =========================

x = token_embedding + position_vectors

print(
    "Final embedding shape:",
    x.shape
)


# =========================
# Create attention model
# =========================

attention = MultiHeadAttention()


# =========================
# Run Multi-Head Attention
# =========================

output = attention(x)