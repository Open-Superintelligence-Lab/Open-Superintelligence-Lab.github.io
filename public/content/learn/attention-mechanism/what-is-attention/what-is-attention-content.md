---
hero:
  title: "What is Attention"
  subtitle: "Understanding the Attention Mechanism"
  tags:
    - "🎯 Attention"
    - "⏱️ 10 min read"
---

Attention lets the model **focus on relevant parts** of the input, just like how you focus on important words when reading!

![Attention Concept](/content/learn/attention-mechanism/what-is-attention/attention-concept.png)

## The Core Idea

**Attention = Weighted average based on relevance**

Instead of treating all inputs equally, attention:
1. Calculates how relevant each input is
2. Weights inputs by relevance
3. Combines them into output

```yaml
Without attention:
  All words matter equally
  "The cat sat on the mat"
  → All words get same weight

With attention:
  Important words matter more
  "The CAT sat on the mat"
  → "cat" gets higher weight
```

## Simple Example

```python
import torch
import torch.nn.functional as F

# Input sequence (3 words, each 4-dim embedding)
sequence = torch.tensor([[0.1, 0.2, 0.3, 0.4],  # word 1
                         [0.5, 0.6, 0.7, 0.8],  # word 2
                         [0.9, 1.0, 1.1, 1.2]]) # word 3

# Attention scores (how important each word is)
attention_weights = torch.tensor([0.1, 0.3, 0.6])  # word 3 most important

# Weighted average
output = torch.zeros(4)
for i, weight in enumerate(attention_weights):
    output += weight * sequence[i]

print(output)
# Mostly influenced by word 3 (weight 0.6)
```

## Query, Key, Value

![QKV Mechanism](/content/learn/attention-mechanism/what-is-attention/qkv-mechanism.png)

Attention uses three concepts:

```yaml
Query (Q): "What am I looking for?"
Key (K): "What do I contain?"
Value (V): "What information do I have?"

Process:
1. Compare Query with all Keys → scores
2. Convert scores to weights (softmax)
3. Weighted sum of Values
```

**Example:**

```python
import torch
import torch.nn.functional as F

# Query: what we're looking for
query = torch.tensor([1.0, 0.0, 1.0])

# Keys: what each position contains
keys = torch.tensor([[1.0, 0.0, 1.0],  # Similar to query!
                     [0.0, 1.0, 0.0],  # Different
                     [1.0, 0.0, 0.8]]) # Somewhat similar

# Values: actual information
values = torch.tensor([[10.0, 20.0],
                       [30.0, 40.0],
                       [50.0, 60.0]])

# 1. Compute attention scores (dot product)
scores = keys @ query
print("Scores:", scores)
# tensor([2.0000, 0.0000, 1.8000])

# 2. Convert to probabilities
weights = F.softmax(scores, dim=0)
print("Weights:", weights)
# tensor([0.5308, 0.0874, 0.3818])

# 3. Weighted sum of values
output = torch.zeros(2)
for i, weight in enumerate(weights):
    output += weight * values[i]

print("Output:", output)
# Mostly from value 0 (weight 0.53)
```

## Why Attention is Powerful

```yaml
Before attention (RNNs):
  Process sequence left-to-right
  Hard to remember distant info
  Slow (sequential)

With attention (Transformers):
  Look at ALL positions at once
  Direct connections everywhere
  Fast (parallel)
  
Result: Better at long sequences!
```

## Self-Attention

**Self-attention: Sequence attends to itself**

```python
# Sentence: "The cat sat"
# Each word attends to all words

"The" attends to: The(0.3), cat(0.2), sat(0.5)
"cat" attends to: The(0.4), cat(0.4), sat(0.2)
"sat" attends to: The(0.1), cat(0.6), sat(0.3)

# Each word builds context from others!
```

## Basic Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        
        # Compute Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (Q.size(-1) ** 0.5)  # Scale
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted values
        output = attn_weights @ V
        
        return output

# Test
attention = SimpleAttention(embed_dim=64)
x = torch.randn(1, 10, 64)  # Batch=1, seq_len=10, dim=64
output = attention(x)
print(output.shape)  # torch.Size([1, 10, 64])
```

## Key Takeaways

✓ **Attention:** Weighted average by relevance

✓ **Q, K, V:** Query, Key, Value mechanism

✓ **Self-attention:** Sequence attends to itself

✓ **Parallel:** Processes all positions at once

✓ **Transformers:** Built entirely on attention

**Remember:** Attention lets models focus on what matters! 🎉
