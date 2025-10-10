---
hero:
  title: "Calculating Attention Scores"
  subtitle: "Computing Query-Key-Value Similarities"
  tags:
    - "🎯 Attention"
    - "⏱️ 10 min read"
---

Attention scores measure **how much each position should attend to every other position**!

![Attention Matrix](/content/learn/attention-mechanism/calculating-attention-scores/attention-matrix.png)

## The Formula

**Score = Q × Kᵀ / √d**

Where:
- Q = Query matrix
- K = Key matrix  
- d = dimension size
- √d = scaling factor

```python
import torch
import torch.nn.functional as F

# Query and Key
Q = torch.randn(1, 10, 64)  # (batch, seq_len, dim)
K = torch.randn(1, 10, 64)

# Compute scores
scores = Q @ K.transpose(-2, -1)  # (1, 10, 10)
scores = scores / (64 ** 0.5)     # Scale by √d

# Convert to probabilities
attn_weights = F.softmax(scores, dim=-1)

print(attn_weights.shape)  # torch.Size([1, 10, 10])
print(attn_weights[0, 0].sum())  # tensor(1.0) ← Sums to 1!
```

## Step-by-Step Example

```python
import torch
import torch.nn.functional as F

# Simple example: 3 positions, 4-dim embeddings
Q = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0, 0.0]])  # (3, 4)

K = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0],
                  [0.5, 0.5, 0.5, 0.5]])  # (3, 4)

# 1. Dot product
scores = Q @ K.T  # (3, 3)
print("Raw scores:")
print(scores)

# 2. Scale
d_k = 4
scaled_scores = scores / (d_k ** 0.5)
print("\\nScaled scores:")
print(scaled_scores)

# 3. Softmax
attn_weights = F.softmax(scaled_scores, dim=-1)
print("\\nAttention weights:")
print(attn_weights)
# Each row sums to 1!
```

## Why Scaling?

```yaml
Without scaling (√d):
  Large dot products → large scores
  Softmax saturates → gradients vanish
  
With scaling:
  Controlled scores
  Stable softmax
  Better gradients
```

## Attention Matrix

```python
# The attention matrix shows who attends to whom
attn_matrix = torch.softmax(Q @ K.T / (d ** 0.5), dim=-1)

print(attn_matrix)
#        Pos 0  Pos 1  Pos 2
# Pos 0 [[0.5,   0.2,   0.3],   ← Position 0 attends to all positions
# Pos 1  [0.1,   0.7,   0.2],   ← Position 1 mostly attends to itself
# Pos 2  [0.4,   0.3,   0.3]]   ← Position 2 attends evenly
```

## Key Takeaways

✓ **Scores:** Measure similarity (dot product)

✓ **Scaling:** Divide by √d for stability

✓ **Softmax:** Convert to probabilities

✓ **Matrix:** Shows all attention connections

**Quick Reference:**

```python
# Compute attention scores
scores = Q @ K.transpose(-2, -1)
scores = scores / (d_k ** 0.5)
attn_weights = F.softmax(scores, dim=-1)

# Apply to values
output = attn_weights @ V
```

**Remember:** Scores tell us where to pay attention! 🎉
