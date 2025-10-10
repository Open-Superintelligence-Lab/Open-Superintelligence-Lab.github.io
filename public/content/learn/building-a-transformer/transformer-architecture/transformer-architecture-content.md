---
hero:
  title: "Transformer Architecture"
  subtitle: "Understanding the Transformer Model"
  tags:
    - "🤖 Transformers"
    - "⏱️ 12 min read"
---

The Transformer is the architecture behind GPT, BERT, and modern LLMs. It's built entirely on attention!

## The Big Picture

**Transformer = Encoder + Decoder (or just one)**

```yaml
Input Text
    ↓
Embedding + Positional Encoding
    ↓
N × Transformer Blocks:
  - Multi-Head Attention
  - Feed-Forward Network
  - Layer Normalization
  - Residual Connections
    ↓
Output Logits
```

## Basic Transformer Block

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Attention block
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)  # Residual connection
        
        # Feedforward block
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)  # Residual connection
        
        return x

# Test
block = TransformerBlock(embed_dim=512, num_heads=8, ff_dim=2048)
x = torch.randn(10, 32, 512)  # (seq, batch, embed)
output = block(x)
print(output.shape)  # torch.Size([10, 32, 512])
```

## Complete Transformer

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, 
                 num_layers=6, ff_dim=2048, max_seq_len=5000):
        super().__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, x):
        batch, seq_len = x.size()
        
        # Token + position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x.transpose(0, 1)).transpose(0, 1)
        
        # Output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

# Create transformer
model = Transformer(vocab_size=50000, num_layers=12)
```

## Key Components

```yaml
1. Embeddings:
   - Token embeddings (vocabulary)
   - Positional embeddings (position info)

2. Transformer Blocks (repeated N times):
   - Multi-head attention
   - Feedforward network
   - Layer normalization
   - Residual connections

3. Output:
   - Final layer norm
   - Linear projection to vocabulary
```

## Key Takeaways

✓ **Self-attention based:** No recurrence, no convolution

✓ **Parallel:** Processes entire sequence at once

✓ **Scalable:** Stack more blocks for more capacity

✓ **Powerful:** Powers GPT, BERT, LLaMA

**Remember:** Transformers are just stacked attention blocks! 🎉
