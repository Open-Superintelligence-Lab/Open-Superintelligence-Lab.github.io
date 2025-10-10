---
hero:
  title: "Attention in Code"
  subtitle: "Complete Attention Implementation"
  tags:
    - "🎯 Attention"
    - "⏱️ 10 min read"
---

Here's the complete, production-ready attention implementation!

## Full Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, heads, seq_len, head_dim)
        
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        output = attn_weights @ V
        
        return output, attn_weights

# Use it
attention = ScaledDotProductAttention()
Q = torch.randn(2, 8, 10, 64)  # batch=2, heads=8, seq=10, dim=64
K = torch.randn(2, 8, 10, 64)
V = torch.randn(2, 8, 10, 64)

output, weights = attention(Q, K, V)
print(output.shape)  # torch.Size([2, 8, 10, 64])
```

## With Masking

```python
# Create causal mask (for autoregressive models)
def create_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # True where we CAN attend

mask = create_causal_mask(5)
print(mask)
# tensor([[ True, False, False, False, False],
#         [ True,  True, False, False, False],
#         [ True,  True,  True, False, False],
#         [ True,  True,  True,  True, False],
#         [ True,  True,  True,  True,  True]])

# Position 0 can only attend to position 0
# Position 1 can attend to positions 0, 1
# etc.
```

## PyTorch Implementation

```python
# Using PyTorch's built-in
attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

x = torch.randn(10, 32, 512)  # (seq, batch, embed)
output, attn_weights = attention(x, x, x)

print(output.shape)  # torch.Size([10, 32, 512])
```

## Key Takeaways

✓ **Complete function:** Q, K, V → Output

✓ **Masking:** Controls what can attend to what

✓ **PyTorch built-in:** Use `nn.MultiheadAttention`

**Remember:** Attention is just a few lines of code! 🎉
