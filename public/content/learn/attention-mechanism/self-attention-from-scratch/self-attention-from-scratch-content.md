---
hero:
  title: "Self Attention from Scratch"
  subtitle: "Building Self-Attention from the Ground Up"
  tags:
    - "🎯 Attention"
    - "⏱️ 10 min read"
---

Let's build self-attention from scratch - the core of transformers!

![Self-Attention Concept](/content/learn/attention-mechanism/self-attention-from-scratch/self-attention-concept.png)

## Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        
        # Project to Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.embed_dim ** 0.5)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply to values
        output = attn_weights @ V
        
        return output

# Test
attention = SelfAttention(embed_dim=64)
x = torch.randn(2, 10, 64)  # Batch=2, seq=10, dim=64
output = attention(x)
print(output.shape)  # torch.Size([2, 10, 64])
```

## Step-by-Step Example

```python
import torch
import torch.nn.functional as F

# Input: 3 words, 4-dim embeddings
x = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0, 0.0]])

# Create Q, K, V projections
W_q = torch.randn(4, 4)
W_k = torch.randn(4, 4)
W_v = torch.randn(4, 4)

# Compute Q, K, V
Q = x @ W_q
K = x @ W_k
V = x @ W_v

# Attention scores
scores = Q @ K.T / (4 ** 0.5)
attn_weights = F.softmax(scores, dim=-1)

# Output
output = attn_weights @ V

print(output.shape)  # torch.Size([3, 4])
```

## Key Takeaways

✓ **Self-attention:** Sequence attends to itself

✓ **Q, K, V:** All come from same input

✓ **Complete implementation:** ~20 lines of code

✓ **Foundation:** Core of transformers

**Remember:** Self-attention is simpler than it looks! 🎉
