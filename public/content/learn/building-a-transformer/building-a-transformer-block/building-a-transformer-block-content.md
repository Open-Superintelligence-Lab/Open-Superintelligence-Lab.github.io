---
hero:
  title: "Building a Transformer Block"
  subtitle: "Creating the Core Transformer Component"
  tags:
    - "🤖 Transformers"
    - "⏱️ 10 min read"
---

A transformer block is the **repeatable unit** that makes transformers work!

![Block Diagram](/content/learn/building-a-transformer/building-a-transformer-block/block-diagram.png)

## The Structure

**Transformer Block = Attention + FFN + Normalization + Residuals**

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 1. Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 2. Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 3. Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Attention sub-block
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))  # Residual + Norm
        
        # FFN sub-block
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # Residual + Norm
        
        return x

# Create and test
block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
x = torch.randn(32, 10, 512)  # (batch, seq, embed)
output = block(x)
print(output.shape)  # torch.Size([32, 10, 512])
```

## The Flow

```yaml
Input
  ↓
Multi-Head Attention
  ↓
Add & Norm (residual connection)
  ↓
Feed-Forward Network
  ↓
Add & Norm (residual connection)
  ↓
Output (same shape as input!)
```

## Residual Connections

**Why residual connections matter:**

```python
# Without residual
output = layer(x)

# With residual
output = x + layer(x)  # Add input back!

# This helps gradients flow during backprop
```

## Stacking Blocks

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Stack N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        
        # Pass through all blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

model = Transformer(vocab_size=50000, n_layers=12)
```

## Key Takeaways

✓ **Core component:** Attention + FFN + Norm + Residuals

✓ **Repeatable:** Stack many blocks

✓ **Same shape:** Input and output dimensions match

✓ **Self-contained:** Each block is independent

**Remember:** Transformers are just stacked blocks! 🎉
