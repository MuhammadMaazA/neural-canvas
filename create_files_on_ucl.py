"""
Helper script to create all training files on UCL machine
Run this on the UCL machine: python3 create_files_on_ucl.py
"""

import os

def create_file(filepath, content):
    """Create a file with given content"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {filepath}")

# Read the actual files
files_to_create = {
    'llm/__init__.py': '''# LLM Package
''',
    
    'llm/transformer_model.py': '''"""
GPT-2 Style Transformer built from scratch
For UCL COMP0220 Coursework - Model 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        Q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x

class GPT2FromScratch(nn.Module):
    """GPT-2 style Transformer trained from scratch"""
    
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12, max_len=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.size()
        
        # Causal mask for autoregressive generation
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(x.device)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.dropout(self.token_embedding(x) + self.position_embedding(positions))
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.fc(x)
        
        # Calculate loss (shift targets for next-token prediction)
        loss = None
        if targets is not None:
            # For next-token prediction: predict tokens[i+1] from tokens[i]
            # Shift logits and targets by 1
            logits = logits[:, :-1, :].contiguous()  # Remove last token
            targets = targets[:, 1:].contiguous()    # Remove first token
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=0)
        
        return logits, loss
''',
}

# Actually, the files are too large to embed here. Let me provide a better solution.
print("This script will help you create files, but files are too large.")
print("Better: Use Git or scp to upload files from your local machine.")
print("\nOr run: git clone <your-repo> on UCL machine if you have a Git repo.")

