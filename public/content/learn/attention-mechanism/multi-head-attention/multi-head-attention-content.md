---
hero:
  title: "Multi-Head Attention"
  subtitle: "Multiple Attention Mechanisms in Parallel"
  tags:
    - "🎯 Attention"
    - "⏱️ 10 min read"
---

Multi-head attention runs **multiple attention mechanisms in parallel**, each focusing on different aspects!

![Multi-Head Visual](/content/learn/attention-mechanism/multi-head-attention/multi-head-visual.png)

## Why Multiple Heads?

Imagine you're reading a sentence. Your brain simultaneously processes:
- **Syntax**: Subject-verb relationships
- **Semantics**: Word meanings and context
- **References**: Pronouns and what they refer to
- **Sentiment**: Emotional tone

Each aspect requires different "attention patterns." Multi-head attention lets the model do this!

### The Core Insight

**One attention head** can only learn one type of relationship pattern.

**Multiple attention heads** can learn different patterns in parallel:
- Head 1: Syntactic dependencies (subject-verb)
- Head 2: Semantic similarity (related concepts)
- Head 3: Positional patterns (nearby words)
- Head 4: Long-range dependencies
- ... etc

### Mathematical Formulation

Instead of:
```
Attention(Q, K, V) → Single output
```

We do:
```
Head₁ = Attention(Q₁, K₁, V₁)
Head₂ = Attention(Q₂, K₂, V₂)
...
Head_h = Attention(Q_h, K_h, V_h)

MultiHead = Concat(Head₁, ..., Head_h) × W_O
```

## The Mechanism

Instead of one attention:
- Run h heads in parallel (typically h=8)
- Each head learns different patterns
- Concatenate outputs
- Project back to original dimension

### Comparing Single vs Multi-Head

```python
import torch
import torch.nn as nn

# Single-head attention: One attention pattern
single_head = nn.MultiheadAttention(embed_dim=512, num_heads=1)
```

**With 1 head:**
- 512-dimensional Q, K, V
- One attention matrix (seq_len × seq_len)
- Can learn one type of relationship

```python
# Multi-head attention: 8 parallel attention patterns!
multi_head = nn.MultiheadAttention(embed_dim=512, num_heads=8)
```

**With 8 heads:**
- Each head gets 512/8 = 64 dimensions
- 8 attention matrices in parallel
- Can learn 8 different relationship types simultaneously

```python
# Test both
x = torch.randn(10, 32, 512)  # (seq_len=10, batch=32, embed_dim=512)

single_output, _ = single_head(x, x, x)
multi_output, _ = multi_head(x, x, x)

print(f"Single head output: {single_output.shape}")  # torch.Size([10, 32, 512])
print(f"Multi-head output: {multi_output.shape}")    # torch.Size([10, 32, 512])
```

**Same output shape!** But multi-head is more expressive.

## Implementing Multi-Head Attention

Let's build it from scratch to understand the internal mechanics:

### Step 1: Initialize Module

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
```

**Dimension split:**
```
embed_dim = 512, num_heads = 8
→ head_dim = 512 / 8 = 64

Each head operates in 64-dimensional space!
```

### Step 2: Create Projection Layers

```python
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_linear = nn.Linear(embed_dim, embed_dim)
```

**Why embed_dim → embed_dim?**
```
Input: 512-dim
Project: 512-dim (but will split into 8 heads of 64-dim each)
Output: 512-dim (after concatenating heads)
```

### Step 3: Forward Pass - Project

```python
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project input to Q, K, V
        Q = self.q_linear(x)  # (batch, seq, embed_dim)
        K = self.k_linear(x)
        V = self.v_linear(x)
```

So far, same as single-head attention.

### Step 4: Split into Multiple Heads

This is the key step!

```python
        # Reshape to split into heads
        # (batch, seq, embed_dim) → (batch, seq, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
```

**The split:**
```
Before: (batch=32, seq=100, embed=512)
After:  (batch=32, seq=100, heads=8, head_dim=64)

We've split 512 dimensions into 8 groups of 64!
```

### Step 5: Transpose for Parallel Processing

```python
        # Rearrange: (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
```

**Why transpose?**
```
We want to process all heads in parallel
Shape (batch, heads, seq, head_dim) allows:
  - Batch dimension stays first
  - Each head is independent
  - Can apply same attention operation to all heads at once
```

### Step 6: Compute Attention for All Heads

```python
        # Attention computation (same for all heads in parallel!)
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = attn @ V
        # Shape: (batch, heads, seq, head_dim)
```

**Magic moment:** This one operation computes attention for ALL 8 heads simultaneously!

```
Each head gets:
  - Its own Q, K, V (64-dim each)
  - Its own attention matrix
  - Its own output

All computed in parallel on GPU!
```

### Step 7: Concatenate Heads

```python
        # Transpose back: (batch, heads, seq, head_dim) → (batch, seq, heads, head_dim)
        output = output.transpose(1, 2).contiguous()
        
        # Flatten heads: (batch, seq, heads, head_dim) → (batch, seq, embed_dim)
        output = output.view(batch_size, seq_len, embed_dim)
```

**The concatenation:**
```
Before: (batch, seq, 8 heads, 64-dim each)
After:  (batch, seq, 512-dim)

We concatenated 8×64 = 512 dimensions back together!
```

### Step 8: Final Projection

```python
        # Mix information from all heads
        output = self.out_linear(output)
        
        return output
```

**Why final projection?** Allows heads to interact and combine their findings!

## Key Takeaways

✓ **Multiple heads:** Each learns different patterns

✓ **Parallel:** All heads run simultaneously

✓ **Standard:** 8 heads is common

**Remember:** More heads = more ways to pay attention! 🎉
