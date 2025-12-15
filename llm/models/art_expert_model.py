\"\"\"
Custom transformer model for art explanation generation.
Uses RoPE, GQA, RMSNorm, and SwiGLU activations.
\"\"\"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    \"\"\"Root Mean Square Normalization.\"\"\"

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - better position encoding"""

    def __init__(self, dim: int, max_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x, seq_len):
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device)
        )


def rotate_half(x):
    """Helper for RoPE"""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to queries and keys"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - more efficient than MHA
    Uses fewer KV heads than Q heads for better memory efficiency
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        assert n_heads % n_kv_heads == 0, f"n_heads {n_heads} must be divisible by n_kv_heads {n_kv_heads}"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads to match Q heads (GQA magic)
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """
    SwiGLU activation - proven better than ReLU/GELU for language models
    Used in PaLM, LLaMA, etc.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: (SiLU(Wx) * Vx)W
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm (more stable training)"""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads, dropout)
        self.feed_forward = SwiGLU(dim, int(dim * 4), dropout)  # 4x expansion ratio
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, mask=None):
        # Pre-norm + residual
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class ArtExpertTransformer(nn.Module):
    """
    Art Expert Transformer - 35-50M parameters

    Perfect balance of:
    - Expressiveness (enough capacity for art knowledge)
    - Efficiency (trains in hours on single GPU)
    - Modern architecture (competitive with larger models)

    Use this as Model 1 (trained from scratch)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,           # Hidden dimension
        n_layers: int = 8,        # Number of layers
        n_heads: int = 8,         # Number of attention heads
        n_kv_heads: int = 2,      # Number of KV heads (GQA)
        max_len: int = 1024,      # Max sequence length
        dropout: float = 0.15,    # Dropout rate
        label_smoothing: float = 0.1  # Label smoothing for better generalization
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.label_smoothing = label_smoothing

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying (shares embeddings with output layer - reduces params)
        self.output.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Initialized ArtExpertTransformer with {total_params/1e6:.1f}M parameters")

    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        """
        Forward pass

        Args:
            x: Input token IDs [batch_size, seq_len]
            targets: Target token IDs for training [batch_size, seq_len]

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Cross-entropy loss (if targets provided)
        """
        batch_size, seq_len = x.shape

        # Create causal mask (prevent attending to future tokens)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(x.device)

        # Embed tokens
        x = self.dropout(self.token_emb(x))

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output projection
        x = self.norm(x)
        logits = self.output(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_targets.view(-1),
                ignore_index=0,  # Ignore padding
                label_smoothing=self.label_smoothing
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ):
        """
        Generate text from prompt

        Args:
            prompt_tokens: Starting tokens [1, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold

        Returns:
            Generated token sequence
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Get logits for last position
            logits, _ = self.forward(prompt_tokens)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            prompt_tokens = torch.cat([prompt_tokens, next_token], dim=1)

            # Stop if EOS token
            if next_token.item() == 0:  # Assuming 0 is EOS
                break

        return prompt_tokens


def create_art_expert_model(vocab_size: int, model_size: str = "base") -> ArtExpertTransformer:
    """
    Factory function to create model with different sizes

    Args:
        vocab_size: Vocabulary size
        model_size: "small" (20M), "base" (35M), or "large" (50M)

    Returns:
        Configured model
    """
    configs = {
        "small": {
            "dim": 384,
            "n_layers": 6,
            "n_heads": 6,
            "n_kv_heads": 2,
            "dropout": 0.15
        },
        "base": {
            "dim": 512,
            "n_layers": 8,
            "n_heads": 8,
            "n_kv_heads": 2,
            "dropout": 0.15
        },
        "large": {
            "dim": 640,
            "n_layers": 10,
            "n_heads": 10,
            "n_kv_heads": 2,
            "dropout": 0.12
        }
    }

    config = configs.get(model_size, configs["base"])

    return ArtExpertTransformer(
        vocab_size=vocab_size,
        **config
    )


if __name__ == "__main__":
    # Test model creation
    print("Testing Art Expert Transformer models...")
    print("=" * 80)

    vocab_size = 50257  # GPT-2 vocab size

    for size in ["small", "base", "large"]:
        model = create_art_expert_model(vocab_size, size)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{size.upper()}: {total_params/1e6:.1f}M parameters")

        # Test forward pass
        x = torch.randint(0, vocab_size, (2, 64))  # batch=2, seq_len=64
        logits, loss = model(x, x)
        print(f"  Output shape: {logits.shape}")
        print()
