---
hero:
  title: "Multi-Head Attention"
  subtitle: "Multiple Attention Mechanisms in Parallel"
  tags:
    - "🎯 Attention"
    - "⏱️ 10 min read"
---

Multi-head attention runs **multiple attention mechanisms in parallel**, each focusing on different aspects!

## The Idea

Instead of one attention:
- Run 8 (or more) attention heads in parallel
- Each head learns different patterns
- Concatenate and project outputs

```python
import torch
import torch.nn as nn

# Single-head attention
single_head = nn.MultiheadAttention(embed_dim=512, num_heads=1)

# Multi-head attention (8 heads)
multi_head = nn.MultiheadAttention(embed_dim=512, num_heads=8)

x = torch.randn(10, 32, 512)  # (seq_len, batch, embed_dim)
output, attn_weights = multi_head(x, x, x)

print(output.shape)  # torch.Size([10, 32, 512])
```

## Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project and split into heads
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention for each head
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = attn @ V
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.out_linear(output)
        
        return output
```

## Key Takeaways

✓ **Multiple heads:** Each learns different patterns

✓ **Parallel:** All heads run simultaneously

✓ **Standard:** 8 heads is common

**Remember:** More heads = more ways to pay attention! 🎉
