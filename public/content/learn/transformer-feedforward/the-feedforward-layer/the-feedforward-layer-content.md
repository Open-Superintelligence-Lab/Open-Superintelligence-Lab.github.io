---
hero:
  title: "The Feedforward Layer"
  subtitle: "FFN in Transformer Blocks"
  tags:
    - "🔀 MoE"
    - "⏱️ 8 min read"
---

The feedforward network (FFN) in transformers processes each position independently!

## The Role of FFN in Transformers

After attention mixes information between positions, the FFN adds **non-linear processing capacity** to each token independently.

### The Transformer Block Pattern

```yaml
For each token:
  1. Self-attention: Mix with other tokens
  2. FFN: Process individually
  3. Repeat N times

Attention: Token interaction
FFN: Individual token transformation
```

**Why both?**
- Attention: "What should I pay attention to?"
- FFN: "What should I do with this information?"

## Mathematical Structure

### The Standard FFN Formula

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
       = W₂ · ReLU(W₁ · x + b₁) + b₂

Where:
  x ∈ ℝ^d_model     (input vector)
  W₁ ∈ ℝ^(d_model × d_ff)   (expand)
  W₂ ∈ ℝ^(d_ff × d_model)   (compress)
  d_ff = 4 × d_model (typical)
```

**Two-step process:**
1. **Expand:** d_model → d_ff (increase dimensionality)
2. **Compress:** d_ff → d_model (reduce back)

### Why Expand Then Compress?

**Information bottleneck and expansion:**
```
Input: 512 dimensions
Expand to: 2048 dimensions  (4× larger)
  → More space for complex transformations
  → Non-linear mixing via ReLU
Compress to: 512 dimensions
  → Extract most important features
```

**Analogy:** Like brainstorming (expand ideas) then summarizing (compress to key points).

## Implementation Step-by-Step

### Step 1: Basic Structure

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
```

**Parameters:**
- `d_model`: Input/output dimension (e.g., 512)
- `d_ff`: Hidden dimension (e.g., 2048 = 4×512)
- `dropout`: Regularization rate

### Step 2: Define Layers

```python
        # Expansion layer
        self.linear1 = nn.Linear(d_model, d_ff)
```

**What it does:**
```
Input: (batch, seq_len, 512)
Matrix multiply: 512 × 2048 weights
Output: (batch, seq_len, 2048)

Each 512-dim vector → 2048-dim vector
```

```python
        # Activation
        self.activation = nn.ReLU()
```

**Non-linearity:** Without this, two linear layers = one linear layer (useless!).

```python
        # Compression layer
        self.linear2 = nn.Linear(d_ff, d_model)
```

**What it does:**
```
Input: (batch, seq_len, 2048)
Matrix multiply: 2048 × 512 weights
Output: (batch, seq_len, 512)

Back to original dimensionality!
```

```python
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
```

### Step 3: Forward Pass

```python
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        
        # Expand
        x = self.linear1(x)      # (batch, seq_len, d_ff)
        
        # Non-linearity
        x = self.activation(x)   # (batch, seq_len, d_ff)
        x = self.dropout(x)
        
        # Compress
        x = self.linear2(x)      # (batch, seq_len, d_model)
        x = self.dropout(x)
        
        return x
```

### Alternative: Sequential API

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
```

**Cleaner but functionally identical!**

## Testing the FFN

```python
# Create FFN
ffn = FeedForward(d_model=512, d_ff=2048, dropout=0.1)

# Test input: 2 sequences of 10 tokens
x = torch.randn(2, 10, 512)

# Forward pass
output = ffn(x)

print(f"Input shape: {x.shape}")     # torch.Size([2, 10, 512])
print(f"Output shape: {output.shape}") # torch.Size([2, 10, 512])

# Shape preserved!
```

**Numerical example for one token:**
```
Input vector (512-dim):
  [0.5, -0.2, 1.3, ..., 0.8]

After expansion (2048-dim):
  [0.3, 0.0, 1.8, 0.0, ..., 1.2]  (many zeros from ReLU)

After compression (512-dim):
  [0.7, -0.1, 1.5, ..., 0.9]  (transformed!)
```

## Parameter Analysis

### Counting Parameters

```python
d_model = 512
d_ff = 2048

# Layer 1: 512 → 2048
params_1 = 512 * 2048 = 1,048,576 (weights)
params_1_bias = 2048 (biases)

# Layer 2: 2048 → 512
params_2 = 2048 * 512 = 1,048,576 (weights)
params_2_bias = 512 (biases)

# Total
total = 1,048,576 + 2,048 + 1,048,576 + 512 = 2,099,712
```

**FFN has ~2.1M parameters for d_model=512!**

### FFN in Full Transformer

```python
# GPT-2 Small
d_model = 768
d_ff = 3072  # 4 × 768
n_layers = 12

# FFN parameters per layer
ffn_params_per_layer = (768 * 3072 + 3072) + (3072 * 768 + 768)
                     = 4,722,432 + 2,360,064
                     = ~7.1M per FFN

# Total FFN parameters
total_ffn_params = 7.1M * 12 layers = ~85M

# Compare to total GPT-2 Small: ~117M parameters
# FFN accounts for ~73% of model parameters!
```

**FFN dominates parameter count in transformers!**

## Modern Variations

### 1. GLU (Gated Linear Units)

Used in modern models like LLaMA:

```python
class GLU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        gate = F.silu(self.gate(x))  # Gating
        up = self.up(x)               # Value
        return self.down(gate * up)   # Element-wise product
```

**Why better:** Gating allows model to learn which features to pass through.

### 2. SwiGLU

```python
# LLaMA uses SwiGLU activation
# Instead of ReLU, uses SiLU (Swish) with gating
activation = nn.SiLU()  # Also called Swish
```

**Formula:**
```
SiLU(x) = x · sigmoid(x)

More smooth than ReLU, better gradients
```

### 3. Expert FFN (MoE)

```python
# Instead of one FFN, use multiple experts
class MoE_FFN(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8):
        super().__init__()
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)
    
    # Route each token to different expert
```

**Enables massive scale with sparse activation!**

## Key Takeaways

✓ **Two layers:** Expand (d_model → d_ff) then compress (d_ff → d_model)

✓ **Position-wise:** Same FFN applied to each token independently

✓ **Standard ratio:** d_ff = 4 × d_model

✓ **Parameter heavy:** FFN contains ~70% of transformer parameters

✓ **Non-linearity:** Critical for model expressiveness

**Remember:** FFN adds individual token processing capacity after attention! 🎉
