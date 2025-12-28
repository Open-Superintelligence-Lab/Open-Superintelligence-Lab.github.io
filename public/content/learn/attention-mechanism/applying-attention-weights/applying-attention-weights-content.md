---
hero:
  title: "Applying Attention Weights"
  subtitle: "Combining Values with Attention"
  tags:
    - "🎯 Attention"
    - "⏱️ 8 min read"
---

After calculating attention weights, we use them to create a **weighted combination of values**!

## The Final Piece of the Puzzle

We've computed attention weights that tell us "who to pay attention to." Now we need to actually USE those weights to retrieve and combine information. This is where the Value matrix comes in!

### The Mathematical Operation

**Output = Attention_Weights × Values**

This is a matrix multiplication that performs a weighted average:

```
AttentionWeights (seq_len × seq_len) @ V (seq_len × dim) = Output (seq_len × dim)
```

**Interpretation:**
- For each query position (row in weights)
- Take weighted combination of all value vectors (columns represent which values)
- Result: context-aware representation

## Step-by-Step Application

Let's trace this carefully with a concrete example:

### Step 1: The Attention Weights

These came from softmax(QK^T/√d):

```python
import torch

# Attention weights (already computed from previous step)
attn_weights = torch.tensor([[0.5, 0.3, 0.2],   # Position 0's attention distribution
                             [0.1, 0.7, 0.2],   # Position 1's attention distribution
                             [0.4, 0.3, 0.3]])  # Position 2's attention distribution
```

**Reading the first row:**
- Position 0 gives 50% attention to position 0
- Position 0 gives 30% attention to position 1
- Position 0 gives 20% attention to position 2

Notice each row sums to 1.0!

### Step 2: The Value Matrix

The values contain the actual information we want to retrieve:

```python
# Values: what information each position has
V = torch.tensor([[1.0, 2.0],   # Position 0's information
                  [3.0, 4.0],   # Position 1's information
                  [5.0, 6.0]])  # Position 2's information
```

Think of values as the "payload" - the actual content we'll extract.

### Step 3: Matrix Multiplication

```python
# One operation computes all weighted combinations!
output = attn_weights @ V

print(output)
# tensor([[2.4000, 3.4000],
#         [3.2000, 4.2000],
#         [2.8000, 3.8000]])
```

**Shape transformation:**
```
(3, 3) @ (3, 2) = (3, 2)
```

Each row of output is a weighted combination of all value vectors!

### Step 4: Manual Verification for Position 0

Let's verify by computing position 0's output by hand:

```yaml
Position 0 output:
  = 0.5 × V[0] + 0.3 × V[1] + 0.2 × V[2]
  = 0.5 × [1.0, 2.0] + 0.3 × [3.0, 4.0] + 0.2 × [5.0, 6.0]
  = [0.5, 1.0] + [0.9, 1.2] + [1.0, 1.2]
  = [2.4, 3.4]
```

**PyTorch output:** [2.4, 3.4] (matches perfectly!)

**What happened:**
- Position 0 mostly retrieves from V[0] (weight 0.5)
- Some from V[1] (weight 0.3)
- A little from V[2] (weight 0.2)

## The Complete Attention Function

Now let's see all three steps together:

Let's implement the complete attention function, broken into clear steps:

### The Complete Function

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V):
    """Complete scaled dot-product attention"""
    
    # STEP 1: Compute similarity scores
    d_k = Q.size(-1)  # Get dimension for scaling
    scores = Q @ K.transpose(-2, -1)
```

**What we have so far:**
- Raw similarity scores between all query-key pairs
- Shape: (batch, seq_len, seq_len)

```python
    # STEP 2: Scale by √d_k
    scores = scores / (d_k ** 0.5)
```

**Why:** Prevents softmax saturation in high dimensions.

```python
    # STEP 3: Convert to probabilities
    attn_weights = F.softmax(scores, dim=-1)
```

**Now:** Each row is a probability distribution summing to 1.

```python
    # STEP 4: Apply to values (weighted average)
    output = attn_weights @ V
    
    return output, attn_weights
```

**Final output:** Context-aware representations!

### Testing the Function

```python
# Create random Q, K, V
Q = torch.randn(1, 5, 64)  # 1 batch, 5 positions, 64-dim
K = torch.randn(1, 5, 64)
V = torch.randn(1, 5, 64)

# Apply attention
output, weights = attention(Q, K, V)

print(f"Input shape: {V.shape}")       # torch.Size([1, 5, 64])
print(f"Output shape: {output.shape}") # torch.Size([1, 5, 64])
print(f"Weights shape: {weights.shape}") # torch.Size([1, 5, 5])
```

**What each output position contains:**
```
Output[0] = weighted combination of all 5 value vectors
Output[1] = weighted combination of all 5 value vectors
...
Output[4] = weighted combination of all 5 value vectors
```

### Inspecting the Attention Weights

```python
print("\\nAttention weights for position 0:")
print(weights[0, 0])
# Example: tensor([0.3, 0.1, 0.4, 0.1, 0.1])
# Position 0 attends mostly to positions 0 (30%) and 2 (40%)

print("\\nWeights sum (should be 1.0):")
print(weights[0, 0].sum())  # tensor(1.0000) ✓
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
