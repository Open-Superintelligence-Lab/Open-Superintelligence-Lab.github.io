---
hero:
  title: "RoPE Positional Encoding"
  subtitle: "Rotary Position Embeddings"
  tags:
    - "🤖 Transformers"
    - "⏱️ 10 min read"
---

RoPE (Rotary Position Embedding) is a modern way to encode position information in transformers!

## Why Position Information is Critical

Transformers process all tokens in parallel - there's no inherent notion of "first" or "last" token. Without position encoding, the model can't distinguish between:

```yaml
"Dog bites man" vs "Man bites dog"
"I love you" vs "You love I"
"The cat" vs "cat The"
```

All would look identical to the transformer! This is a fundamental problem we must solve.

### The Evolution of Positional Encoding

**Original Transformer (2017):** Sinusoidal positional encoding
```
Fixed sine/cosine functions
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**BERT/GPT-2:** Learned positional embeddings
```
Train position embeddings just like token embeddings
Problem: Fixed maximum sequence length
```

**RoPE (2021):** Rotary positional embedding
```
Encodes relative positions through rotation
Used in: LLaMA, PaLM, GPT-NeoX, many modern LLMs
Benefits: Extrapolates to longer sequences!
```

## The Mathematical Foundation of RoPE

RoPE encodes position information by **rotating** the query and key vectors in embedding space. This is based on complex number rotation!

### The Core Idea: Rotation Matrices

In 2D, rotating a vector by angle θ:
```
[x']   [cos(θ)  -sin(θ)] [x]
[y'] = [sin(θ)   cos(θ)] [y]
```

**RoPE applies this to pairs of dimensions** in Q and K vectors!

### Why Rotation?

**Key mathematical property:**
```
After rotation by θ₁ and θ₂:
  dot_product(rotate(q, θ₁), rotate(k, θ₂)) 
  = function of (θ₁ - θ₂)
  = function of relative position!
```

This means attention scores depend on **relative** positions, not absolute ones!

## How RoPE Works: Step-by-Step

### Step 1: Compute Rotation Frequencies

```python
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        # Compute inverse frequencies for rotation
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
```

**What are inverse frequencies?**
```
For dimension pairs (0,1), (2,3), (4,5), etc.:
  freq[0] = 1 / 10000^(0/dim) = 1.0
  freq[1] = 1 / 10000^(2/dim) ≈ 0.954
  freq[2] = 1 / 10000^(4/dim) ≈ 0.912
  ...
  freq[dim/2-1] ≈ 0.0001

Different frequencies for different dimension pairs!
```

**Why different frequencies?**
- Low-frequency rotations: Encode long-range patterns
- High-frequency rotations: Encode local patterns
- Together: Capture multi-scale positional information

### Step 2: Compute Position-Dependent Angles

```python
    def forward(self, x):
        seq_len = x.size(1)
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # Outer product: position × frequency
        freqs = torch.outer(t, self.inv_freq)
        # Shape: (seq_len, dim/2)
```

**What's happening:**
```
For position p and dimension pair d:
  angle[p, d] = p × inv_freq[d]

Example with 3 positions, 4 dims (2 pairs):
  angles[0] = [0×1.0,  0×0.9]  = [0.0, 0.0]
  angles[1] = [1×1.0,  1×0.9]  = [1.0, 0.9]
  angles[2] = [2×1.0,  2×0.9]  = [2.0, 1.8]

Position 2 has larger rotation angles than position 1!
```

### Step 3: Create Sine and Cosine Components

```python
        # Duplicate for full dimensions
        emb = torch.cat((freqs, freqs), dim=-1)
        # Shape: (seq_len, dim)
        
        # Compute cos and sin
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        return cos_emb, sin_emb
```

These will be used to rotate the vectors!

### Step 4: Apply Rotation

Now the actual rotation happens:

```python
def apply_rope(x, cos, sin):
    """Apply rotary embeddings to vector x"""
    # Split into even and odd dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]
    # x1 = dimensions [0, 2, 4, 6, ...]
    # x2 = dimensions [1, 3, 5, 7, ...]
```

**Why split even/odd?** We rotate pairs of dimensions together!

```python
    # Apply 2D rotation formula
    rotated = torch.cat([
        x1 * cos - x2 * sin,  # Rotated even dims
        x1 * sin + x2 * cos   # Rotated odd dims
    ], dim=-1)
    
    return rotated
```

**The rotation formula:**
```
For each dimension pair (x₁, x₂):
  x₁' = x₁ × cos(θ) - x₂ × sin(θ)
  x₂' = x₁ × sin(θ) + x₂ × cos(θ)

This is the standard 2D rotation matrix applied!
```

### Step 5: Using RoPE

```python
# Create RoPE module
rope = RotaryPositionalEmbedding(dim=64)

# Input embeddings
x = torch.randn(1, 10, 64)  # 1 batch, 10 tokens, 64-dim

# Get rotation matrices
cos, sin = rope(x)

# Apply rotation
x_with_pos = apply_rope(x, cos, sin)

print(f"Original shape: {x.shape}")          # torch.Size([1, 10, 64])
print(f"Rotated shape: {x_with_pos.shape}")  # torch.Size([1, 10, 64])
```

**What changed:**
- Shape: Same!
- Values: Rotated based on position
- Position 0: Small rotation
- Position 9: Larger rotation

### Understanding the Math

**For position p and dimension pair (i, i+1):**
```
θ = p / 10000^(2i/d)

Rotation:
  [q_i  ]   [cos(θ)  -sin(θ)] [q_i  ]
  [q_i+1] = [sin(θ)   cos(θ)] [q_i+1]

As p increases → θ increases → more rotation
```

**Example numerical walkthrough:**
```
Position 0, dim pair (0,1), freq=1.0:
  θ = 0 × 1.0 = 0
  cos(0) = 1, sin(0) = 0
  → No rotation (position 0 is reference)

Position 5, dim pair (0,1), freq=1.0:
  θ = 5 × 1.0 = 5 radians
  cos(5) ≈ 0.28, sin(5) ≈ -0.96
  → Significant rotation!
```

## Why RoPE is Superior

Let's compare the three approaches:

### 1. Learned Positional Embeddings (GPT-2, BERT)

```python
pos_embedding = nn.Embedding(max_seq_len, d_model)
x = token_emb + pos_embedding(positions)
```

**Problems:**
```yaml
✗ Fixed maximum length (e.g., 2048 tokens)
✗ Can't extrapolate to longer sequences
✗ Absolute positions only
✗ No relative position information

Example:
  Trained on max_len=512
  Test on seq_len=1024 → FAILS (no embedding for pos > 512)
```

### 2. Sinusoidal Positional Encoding (Original Transformer)

```python
# Fixed sine/cosine functions
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

x = token_emb + PE
```

**Better:**
```yaml
✓ Works for any sequence length
✗ Still absolute positions
✗ Added to embeddings (not ideal)
```

### 3. RoPE (Modern LLMs)

```python
# Rotate Q and K based on position
Q_rotated = apply_rope(Q, position)
K_rotated = apply_rope(K, position)
attention = Q_rotated @ K_rotated^T
```

**Best:**
```yaml
✓ Works for any sequence length
✓ Encodes RELATIVE positions
✓ Applied to Q,K (not added to embeddings)
✓ Better extrapolation beyond training length
✓ More parameter efficient

Models using RoPE:
  - LLaMA 1 & 2 (Meta)
  - PaLM (Google)
  - GPT-NeoX (EleutherAI)
  - Falcon
  - Most new open-source LLMs
```

### Mathematical Advantage: Relative Positions

**The key property:**
```
dot_product(RoPE(q, m), RoPE(k, n)) = f(q, k, m-n)

Where:
  m, n = positions
  m-n = relative distance

The attention score depends on RELATIVE position (m-n), not absolute!
```

**Why this matters:**
```
Token at position 10 attending to position 5:
  Relative distance = 10-5 = 5

Token at position 100 attending to position 95:
  Relative distance = 100-95 = 5

Same relative distance → Same positional bias!
This allows generalization beyond training length.
```

### Extrapolation Example

```python
# Train on sequences up to length 512
model_learned_emb = TransformerWithLearnedPos(max_len=512)
# Test on length 1024 → Poor performance

# Train on sequences up to length 512
model_rope = TransformerWithRoPE()
# Test on length 1024 → Good performance!

# RoPE extrapolates because it learned relative patterns
```

## Key Takeaways

✓ **Rotary:** Encodes position via rotation

✓ **Relative:** Captures relative positions

✓ **Modern:** Used in latest LLMs

**Remember:** RoPE is the modern way to handle positions! 🎉
