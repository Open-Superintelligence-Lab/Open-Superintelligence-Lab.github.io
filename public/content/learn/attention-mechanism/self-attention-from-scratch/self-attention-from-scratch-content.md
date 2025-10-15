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

## What Makes Self-Attention "Self"?

**Self-attention** means the input sequence attends to **itself**. Unlike cross-attention (where one sequence attends to another), here Q, K, and V all come from the same source.

**Think of it like:** A group of friends (the sequence) where each person (position) considers input from everyone in the group (including themselves) to update their understanding.

### The Mathematical Structure

```
Input: X (seq_len × embed_dim)
  ↓
Q = XW_Q  (transform for querying)
K = XW_K  (transform for keys)
V = XW_V  (transform for values)
  ↓
Attention(Q, K, V) = softmax(QK^T/√d)V
  ↓
Output: (seq_len × embed_dim)
```

Notice: X is the source for Q, K, AND V!

## Building Self-Attention Step-by-Step

Let's implement this piece by piece to understand each component:

### Step 1: Initialize the Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
```

### Step 2: Create Q, K, V Projections

```python
        # Three learned linear transformations
        # All take embed_dim → embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
```

**Why separate projections?**
- Each learns a different "view" of the input
- Query learns: "What patterns am I looking for?"
- Key learns: "What patterns do I represent?"
- Value learns: "What information should I provide?"

Even though they start from the same input (X), the learned transformations make them serve different roles!

### Step 3: The Forward Pass

```python
    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        # Example: (32, 100, 512) = 32 sentences, 100 tokens, 512-dim
```

**Step 3a: Project Input to Q, K, V**

```python
        # All three come from the same input X!
        Q = self.query(x)  # (batch, seq_len, embed_dim)
        K = self.key(x)    # (batch, seq_len, embed_dim)
        V = self.value(x)  # (batch, seq_len, embed_dim)
```

This is the "self" part - x generates all three!

**Step 3b: Compute Similarity Scores**

```python
        # Matrix multiplication: Q @ K^T
        scores = Q @ K.transpose(-2, -1)
        # Shape: (batch, seq_len, seq_len)
        # scores[i,j] = similarity between position i and j
```

**Step 3c: Scale for Stability**

```python
        # Scale by square root of dimension
        scores = scores / (self.embed_dim ** 0.5)
```

Prevents softmax saturation!

**Step 3d: Softmax to Get Weights**

```python
        # Convert scores to probability distribution
        attn_weights = F.softmax(scores, dim=-1)
        # Each row sums to 1.0
```

**Step 3e: Apply to Values**

```python
        # Weighted average of all values
        output = attn_weights @ V
        
        return output
```

### Step 4: Testing the Module

```python
# Create self-attention layer
attention = SelfAttention(embed_dim=64)

# Input: 2 sequences, 10 tokens each, 64-dimensional
x = torch.randn(2, 10, 64)

# Apply self-attention
output = attention(x)

print(f"Input shape: {x.shape}")      # torch.Size([2, 10, 64])
print(f"Output shape: {output.shape}") # torch.Size([2, 10, 64])
```

**What happened:**
- Each of the 10 tokens now contains context from all other tokens
- The network learned (through W_Q, W_K, W_V) what context is relevant
- All done in one parallel operation!

## Working Through a Manual Example

Let's see self-attention with tiny numbers we can verify by hand:

### Step 1: Input Sequence

```python
import torch
import torch.nn.functional as F

# Input: 3 word embeddings, 4 dimensions each
x = torch.tensor([[1.0, 0.0, 1.0, 0.0],  # Word 1: "The"
                  [0.0, 1.0, 0.0, 1.0],  # Word 2: "cat"
                  [1.0, 1.0, 0.0, 0.0]]) # Word 3: "sat"

print(f"Input shape: {x.shape}")  # torch.Size([3, 4])
```

This is our starting point - the sentence we want to process.

### Step 2: Create Projection Matrices

```python
# Initialize random weight matrices
# In practice, these are learned during training
W_q = torch.randn(4, 4)  # Query projection
W_k = torch.randn(4, 4)  # Key projection
W_v = torch.randn(4, 4)  # Value projection
```

**Why random?** These would normally be learned, but for this example we'll use random weights to show the mechanism.

### Step 3: Project to Q, K, V

```python
# Transform input into Q, K, V
# Notice: all three come from the SAME input x!
Q = x @ W_q  # Shape: (3, 4)
K = x @ W_k  # Shape: (3, 4)
V = x @ W_v  # Shape: (3, 4)

print(f"Q shape: {Q.shape}")  # torch.Size([3, 4])
print(f"K shape: {K.shape}")  # torch.Size([3, 4])
print(f"V shape: {V.shape}")  # torch.Size([3, 4])
```

**The self-attention key:** Even though Q, K, V have different values (due to different projections), they all originated from x!

### Step 4: Compute Attention Scores

```python
# Similarity between all query-key pairs
d_k = 4
scores = Q @ K.T / (d_k ** 0.5)

print(f"Scores shape: {scores.shape}")  # torch.Size([3, 3])
print("Scores:")
print(scores)
```

**Scores matrix:**
```
         Key0  Key1  Key2
Query0 [[s00,  s01,  s02],   ← How "The" relates to all words
Query1  [s10,  s11,  s12],   ← How "cat" relates to all words
Query2  [s20,  s21,  s22]]   ← How "sat" relates to all words
```

### Step 5: Softmax to Get Weights

```python
attn_weights = F.softmax(scores, dim=-1)

print("Attention weights:")
print(attn_weights)
print(f"\\nRow 0 sum: {attn_weights[0].sum()}")  # 1.0
```

**Now each row is a probability distribution!**

### Step 6: Apply to Values

```python
# Weighted combination of all value vectors
output = attn_weights @ V

print(f"\\nOutput shape: {output.shape}")  # torch.Size([3, 4])
```

**Final result:**
- Each word now has a new representation
- This representation includes context from all other words
- The attention weights determined how much context from each word

### Understanding the Output

```
Original "cat" representation: [0.0, 1.0, 0.0, 1.0]

After self-attention "cat" representation: [0.X, 0.Y, 0.Z, 0.W]
  Contains weighted info from:
  - "The" (e.g., 40% weight)
  - "cat" itself (e.g., 40% weight)
  - "sat" (e.g., 20% weight)

Now "cat" knows it's in context of "The _ sat"!
```

## Key Takeaways

✓ **Self-attention:** Sequence attends to itself

✓ **Q, K, V:** All come from same input

✓ **Complete implementation:** ~20 lines of code

✓ **Foundation:** Core of transformers

**Remember:** Self-attention is simpler than it looks! 🎉
