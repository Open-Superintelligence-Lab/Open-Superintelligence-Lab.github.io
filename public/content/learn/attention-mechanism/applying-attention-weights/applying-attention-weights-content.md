---
hero:
  title: "Applying Attention Weights"
  subtitle: "Combining Values with Attention"
  tags:
    - "🎯 Attention"
    - "⏱️ 8 min read"
---

After calculating attention weights, we use them to create a **weighted combination of values**!

## The Final Step

**Output = Attention_Weights × Values**

```python
import torch

# Attention weights (from softmax)
attn_weights = torch.tensor([[0.5, 0.3, 0.2],   # Position 0 attends to...
                             [0.1, 0.7, 0.2],   # Position 1 attends to...
                             [0.4, 0.3, 0.3]])  # Position 2 attends to...

# Values (what information each position has)
V = torch.tensor([[1.0, 2.0],   # Position 0 value
                  [3.0, 4.0],   # Position 1 value
                  [5.0, 6.0]])  # Position 2 value

# Apply attention
output = attn_weights @ V

print(output)
# tensor([[2.2000, 3.2000],
#         [2.8000, 3.8000],
#         [2.6000, 3.6000]])
```

**Manual calculation for position 0:**

```yaml
Position 0 output:
  = 0.5 × [1.0, 2.0] + 0.3 × [3.0, 4.0] + 0.2 × [5.0, 6.0]
  = [0.5, 1.0] + [0.9, 1.2] + [1.0, 1.2]
  = [2.4, 3.4]
  
This is a weighted average!
```

## Complete Attention

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V):
    """Complete attention mechanism"""
    # 1. Compute scores
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    
    # 2. Softmax to get weights
    attn_weights = F.softmax(scores, dim=-1)
    
    # 3. Apply to values
    output = attn_weights @ V
    
    return output, attn_weights

# Test
Q = torch.randn(1, 5, 64)
K = torch.randn(1, 5, 64)
V = torch.randn(1, 5, 64)

output, weights = attention(Q, K, V)
print(output.shape)  # torch.Size([1, 5, 64])
```

## Key Takeaways

✓ **Final step:** Multiply attention weights by values

✓ **Weighted average:** Combines information by relevance

✓ **Output:** Context-aware representation

**Quick Reference:**

```python
# Attention output
output = attention_weights @ V
```

**Remember:** Attention weights select which values to use! 🎉
