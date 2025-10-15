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

## Understanding Attention Scores

Attention scores are the **heart** of the attention mechanism. They determine which parts of the input are relevant for each output position.

### The Core Computation

Attention scores answer the question: "How similar is my query to each key?"

**Mathematical definition:**
```
Scores = QK^T / √d_k

Where:
  Q = Query matrix (seq_len × d_k)
  K = Key matrix (seq_len × d_k)  
  K^T = Key transposed (d_k × seq_len)
  d_k = dimension of keys
  √d_k = scaling factor
```

**Result:** A score matrix (seq_len × seq_len) where:
- Rows represent query positions (who is asking)
- Columns represent key positions (what they're asking about)
- Each element is the similarity score

## The Formula Breakdown

**Score = Q × K^T / √d**

Let's understand each component:

### Component 1: Q (Query Matrix)

```python
import torch
import torch.nn.functional as F

# Query: What each position is looking for
Q = torch.randn(1, 10, 64)  # (batch=1, seq_len=10, dim=64)
```

**Shape explanation:**
- Batch=1: One sequence at a time
- Seq_len=10: 10 positions (words/tokens)
- Dim=64: Each query is 64-dimensional

### Component 2: K (Key Matrix)

```python
# Keys: What each position represents
K = torch.randn(1, 10, 64)  # Same shape as Q
```

Q and K must have the same dimension for dot product!

### Component 3: Matrix Multiplication QK^T

```python
# Compute all pairwise dot products
scores = Q @ K.transpose(-2, -1)
print(scores.shape)  # torch.Size([1, 10, 10])
```

**What this does:**
```
Q (1, 10, 64) @ K^T (1, 64, 10) = Scores (1, 10, 10)

Scores[i, j] = dot product of query[i] and key[j]
             = how well query at position i matches key at position j
```

**The attention matrix:** Each position now has a score for every other position!

### Component 4: Scaling by √d

```python
# Scale to prevent large values
d_k = 64
scores = scores / (d_k ** 0.5)  # Divide by √64 = 8
```

**Why scaling is critical:**

Without scaling:
```
64-dim dot product can be large: range [-64, 64]
Softmax of large values saturates: almost all weight to max
Gradient ≈ 0 (vanishing gradients!)
```

With scaling:
```
Scaled scores in reasonable range: ~ [-8, 8]
Softmax distributes weights better
Gradients flow well!
```

### Component 5: Softmax to Get Weights

```python
# Convert scores to probability distribution
attn_weights = F.softmax(scores, dim=-1)

print(attn_weights.shape)        # torch.Size([1, 10, 10])
print(attn_weights[0, 0].sum())  # tensor(1.0) ← Each row sums to 1!
```

**What softmax does:**
- Converts raw scores to probabilities (0 to 1)
- Each row sums to exactly 1.0
- Higher scores get higher probabilities (exponentially!)

## Working Through a Complete Example

Let's trace the computation with actual numbers:

### Step 1: Create Small Example Data

```python
import torch
import torch.nn.functional as F

# 3 positions (words), 4-dimensional embeddings
Q = torch.tensor([[1.0, 0.0, 1.0, 0.0],   # Query at position 0
                  [0.0, 1.0, 0.0, 1.0],   # Query at position 1
                  [1.0, 1.0, 0.0, 0.0]])  # Query at position 2

K = torch.tensor([[1.0, 0.0, 1.0, 0.0],   # Key at position 0
                  [0.0, 1.0, 0.0, 1.0],   # Key at position 1
                  [0.5, 0.5, 0.5, 0.5]])  # Key at position 2
```

### Step 2: Compute Pairwise Dot Products

```python
# Matrix multiplication: Q @ K^T
scores = Q @ K.T  # Shape: (3, 3)
print("Raw scores:")
print(scores)
```

**Manual calculation for position 0:**
```
Position 0 query · Position 0 key:
  [1,0,1,0] · [1,0,1,0] = 1+0+1+0 = 2.0

Position 0 query · Position 1 key:
  [1,0,1,0] · [0,1,0,1] = 0+0+0+0 = 0.0

Position 0 query · Position 2 key:
  [1,0,1,0] · [0.5,0.5,0.5,0.5] = 0.5+0+0.5+0 = 1.0
```

**Full scores matrix:**
```
       Key0  Key1  Key2
Query0 [2.0,  0.0,  1.0]
Query1 [0.0,  2.0,  1.0]
Query2 [1.0,  1.0,  1.0]
```

### Step 3: Scale by √d

```python
d_k = 4
scaled_scores = scores / (d_k ** 0.5)  # Divide by √4 = 2
print("\\nScaled scores:")
print(scaled_scores)
```

**After scaling:**
```
       Key0  Key1  Key2
Query0 [1.0,  0.0,  0.5]
Query1 [0.0,  1.0,  0.5]
Query2 [0.5,  0.5,  0.5]
```

Now the values are in a more reasonable range!

### Step 4: Apply Softmax

```python
attn_weights = F.softmax(scaled_scores, dim=-1)
print("\\nAttention weights:")
print(attn_weights)
```

**After softmax (each row sums to 1):**
```
         Pos0   Pos1   Pos2
Query0 [0.576, 0.212, 0.212]  ← Mostly attends to position 0
Query1 [0.212, 0.576, 0.212]  ← Mostly attends to position 1
Query2 [0.333, 0.333, 0.333]  ← Attends equally to all
```

### Understanding the Result

**Position 0:** 
- Query matched Key0 best (score 2.0 before scaling)
- After softmax: 57.6% attention to position 0

**Position 2:**
- Query matched all keys equally (scores all 1.0)
- After softmax: Equal attention (33.3% each)

This makes intuitive sense!

## The Mathematical Necessity of Scaling

Scaling by √d is not optional - it's essential for training stability!

### The Problem Without Scaling

**Mathematical analysis:**

For d-dimensional vectors with random values:
```
Expected value of dot product: E[q · k] = 0
Variance of dot product: Var[q · k] = d
Standard deviation: σ = √d
```

**Example with d=64:**
```
Dot products can range from approximately:
  [-3√64, 3√64] = [-24, 24]  (3 standard deviations)

After softmax:
  e^24 / (e^24 + ...) ≈ 0.9999999...
  e^(-24) / (...) ≈ 0.0000001...
```

The softmax becomes extremely sharp - almost all weight to the max score!

### The Solution: Scaling

**Divide by √d:**
```
Scaled scores: (-24/8, 24/8) = [-3, 3]

After softmax:
  e^3 / (e^3 + ...) ≈ 0.95  (still high but not saturated)
  e^(-3) / (...) ≈ 0.05     (small but has gradient)
```

**Impact on gradients:**
```
Without scaling:
  Softmax saturated → gradient ≈ 0 → no learning ✗

With scaling:
  Softmax smooth → gradient flows → good learning ✓
```

### Empirical Evidence

```python
import torch
import torch.nn.functional as F

# High-dimensional vectors
Q = torch.randn(1, 10, 256)
K = torch.randn(1, 10, 256)

# Without scaling
scores_unscaled = Q @ K.transpose(-2, -1)
weights_unscaled = F.softmax(scores_unscaled, dim=-1)

# With scaling
scores_scaled = (Q @ K.transpose(-2, -1)) / (256 ** 0.5)
weights_scaled = F.softmax(scores_scaled, dim=-1)

print("Without scaling - max weight:", weights_unscaled[0,0].max().item())
# ~0.999 (saturated!)

print("With scaling - max weight:", weights_scaled[0,0].max().item())
# ~0.45 (distributed!)
```

Scaling makes attention weights much better distributed!

## The Attention Matrix Visualization

The attention matrix is a powerful tool for understanding what your model learned:

### Creating the Matrix

```python
# Compute full attention matrix
Q = torch.randn(3, 4)  # 3 positions, 4-dim
K = torch.randn(3, 4)
d = 4

attn_matrix = torch.softmax(Q @ K.T / (d ** 0.5), dim=-1)

print("Attention Matrix:")
print(attn_matrix)
```

**Example output:**
```
Attention Matrix:
         Pos0   Pos1   Pos2
Pos0 [[0.500, 0.200, 0.300],   ← Position 0's attention distribution
Pos1  [0.100, 0.700, 0.200],   ← Position 1's attention distribution
Pos2  [0.400, 0.300, 0.300]]   ← Position 2's attention distribution
```

### Reading the Matrix

**Row interpretation (who is attending):**
- Row 0: How position 0 distributes its attention
- Row 1: How position 1 distributes its attention  
- Row 2: How position 2 distributes its attention

**Column interpretation (who is being attended to):**
- Column 0: How much total attention position 0 receives
- Column 1: How much total attention position 1 receives
- Column 2: How much total attention position 2 receives

**Element interpretation:**
```
Matrix[i,j] = How much position i attends to position j

Example:
  Matrix[1,1] = 0.7
  → Position 1 gives 70% of its attention to itself!
  → Strong self-attention
```

### Real-World Interpretation

For sentence "The cat sat":

```
          "The"  "cat"  "sat"
"The"   [[0.2,   0.3,   0.5],   ← "The" focuses on "sat"
"cat"    [0.4,   0.4,   0.2],   ← "cat" focuses on itself + "The"
"sat"    [0.1,   0.6,   0.3]]   ← "sat" focuses on "cat" (the subject!)
```

This makes linguistic sense - "sat" should focus on who's doing the sitting!

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
