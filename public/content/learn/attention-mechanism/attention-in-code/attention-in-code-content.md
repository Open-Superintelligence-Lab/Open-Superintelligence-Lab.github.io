---
hero:
  title: "Attention in Code"
  subtitle: "Complete Attention Implementation"
  tags:
    - "🎯 Attention"
    - "⏱️ 10 min read"
---

Here's the complete, production-ready attention implementation!

## From Theory to Production Code

We've learned the theory. Now let's build production-quality attention with all the features used in real transformers:
- Dropout for regularization
- Masking for causal attention
- Efficient batched computation

## Building Production Attention

Let's build this incrementally to understand each feature:

### Step 1: Basic Class Structure

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
```

**Why dropout?** Prevents overfitting by randomly zeroing some attention weights during training.

### Step 2: Forward Pass - Compute Scores

```python
    def forward(self, Q, K, V, mask=None):
        # Input shapes: (batch, heads, seq_len, head_dim)
        # Example: (32, 8, 100, 64)
        
        # Get dimension for scaling
        d_k = Q.size(-1)  # head_dim, typically 64
        
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1)
        # Shape: (batch, heads, seq_len, seq_len)
```

**What we have:** Raw similarity scores for all position pairs.

### Step 3: Scale Scores

```python
        # Scale by √d_k
        scores = scores / (d_k ** 0.5)
```

Critical for numerical stability!

### Step 4: Apply Masking (Optional)

```python
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
```

**What masking does:**
- Sets masked positions to -∞
- After softmax, -∞ becomes 0
- Effectively blocks attention to certain positions

**Use cases:**
- Padding tokens: Don't attend to padding
- Causal masking: Can't see future tokens (GPT-style)

### Step 5: Softmax and Dropout

```python
        # Convert to probabilities
        attn_weights = F.softmax(scores, dim=-1)
        # After masking, masked positions have weight 0
        
        # Apply dropout (only during training)
        attn_weights = self.dropout(attn_weights)
```

Dropout randomly zeros some weights for regularization.

### Step 6: Apply to Values

```python
        # Weighted combination of values
        output = attn_weights @ V
        
        return output, attn_weights
```

Returns both output and attention weights (useful for visualization).

### Step 7: Using the Module

```python
# Create attention module
attention = ScaledDotProductAttention(dropout=0.1)

# Prepare inputs (typical transformer dimensions)
Q = torch.randn(2, 8, 10, 64)  # batch=2, heads=8, seq=10, dim=64
K = torch.randn(2, 8, 10, 64)
V = torch.randn(2, 8, 10, 64)

# Apply attention
output, weights = attention(Q, K, V)

print(f"Output shape: {output.shape}")   # torch.Size([2, 8, 10, 64])
print(f"Weights shape: {weights.shape}") # torch.Size([2, 8, 10, 10])
```

**Shape breakdown:**
- Batch: 2 sequences
- Heads: 8 parallel attention mechanisms
- Sequence: 10 tokens
- Dimension: 64 per head
- Weights: 10×10 attention matrix per head

## Understanding Masking

Masking is crucial for many attention applications. Let's explore it:

### Creating a Causal Mask

For autoregressive models (like GPT), we need **causal masking** - positions can only attend to previous positions, not future ones!

```python
def create_causal_mask(seq_len):
    """Create mask where position i can only attend to positions ≤ i"""
    # Create upper triangular matrix of 1s
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    # Convert: 1 → False (can't attend), 0 → True (can attend)
    return mask == 0

# Create mask for sequence length 5
mask = create_causal_mask(5)
print("Causal Mask:")
print(mask)
```

**Output:**
```
Causal Mask:
       Pos0  Pos1  Pos2  Pos3  Pos4
Pos0 [[ T,    F,    F,    F,    F],    ← Can only see position 0 (itself)
Pos1  [ T,    T,    F,    F,    F],    ← Can see positions 0-1
Pos2  [ T,    T,    T,    F,    F],    ← Can see positions 0-2
Pos3  [ T,    T,    T,    T,    F],    ← Can see positions 0-3
Pos4  [ T,    T,    T,    T,    T]]    ← Can see all positions (0-4)

T = True (can attend), F = False (cannot attend)
```

**Why this pattern?**
```
When generating text:
  Position 0 (first word): No context yet
  Position 1 (second word): Can use first word as context
  Position 2 (third word): Can use first two words
  ...
  
This prevents "looking into the future"!
```

### Using the Mask

```python
# Apply attention with causal mask
seq_len = 5
mask = create_causal_mask(seq_len)

Q = torch.randn(1, 1, seq_len, 64)
K = torch.randn(1, 1, seq_len, 64)
V = torch.randn(1, 1, seq_len, 64)

output, weights = attention(Q, K, V, mask=mask)

print("Attention weights with masking:")
print(weights[0, 0])
# Notice upper triangle is all zeros!
```

**Example weights with causal mask:**
```
[[1.00, 0.00, 0.00, 0.00, 0.00],   ← Pos 0: only itself
 [0.40, 0.60, 0.00, 0.00, 0.00],   ← Pos 1: 0 and 1
 [0.20, 0.30, 0.50, 0.00, 0.00],   ← Pos 2: 0, 1, and 2
 [0.10, 0.20, 0.30, 0.40, 0.00],   ← Pos 3: 0-3
 [0.15, 0.20, 0.25, 0.25, 0.15]]   ← Pos 4: all positions
```

### Padding Mask

Another common mask type - ignore padding tokens:

```python
def create_padding_mask(seq, pad_token=0):
    """Mask out padding tokens"""
    # True where NOT padding
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

# Example sequence with padding
seq = torch.tensor([[1, 2, 3, 0, 0],   # Real: 3 tokens, Padding: 2
                    [1, 2, 3, 4, 5]])  # Real: 5 tokens, No padding

mask = create_padding_mask(seq)
# Don't attend to positions with token_id=0
```

## Using PyTorch's Built-In Attention

For production, use PyTorch's optimized implementation:

```python
# PyTorch's MultiheadAttention
attention = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1
)

# Input shape: (seq_len, batch, embed_dim)  ← Note: seq first!
x = torch.randn(10, 32, 512)  # 10 tokens, batch 32, 512-dim

# Self-attention: same input for Q, K, V
output, attn_weights = attention(x, x, x)

print(output.shape)  # torch.Size([10, 32, 512])
```

**Important:** PyTorch expects (seq, batch, embed) format, not (batch, seq, embed)!

## Key Takeaways

✓ **Complete function:** Q, K, V → Output

✓ **Masking:** Controls what can attend to what

✓ **PyTorch built-in:** Use `nn.MultiheadAttention`

**Remember:** Attention is just a few lines of code! 🎉
