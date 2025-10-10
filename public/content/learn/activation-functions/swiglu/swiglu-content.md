---
hero:
  title: "SwiGLU"
  subtitle: "Swish-Gated Linear Unit - Advanced Activation"
  tags:
    - "⚡ Activation Functions"
    - "⏱️ 10 min read"
---

SwiGLU is a **gated activation function** used in state-of-the-art language models like LLaMA and PaLM. It's more complex than ReLU but much more powerful!

## The Concept: Gating

**Gating = One path controls another path**

Think of it like a smart light switch - one signal decides how much of another signal gets through!

![SwiGLU Architecture](/content/learn/activation-functions/swiglu/swiglu-architecture.png)

## The Formula

**SwiGLU(x) = SiLU(W₁(x)) ⊙ V(x)**

Where:
- `W₁(x)` = first linear transformation
- `SiLU()` = activation (swish)
- `V(x)` = second linear transformation (gate)
- `⊙` = element-wise multiplication

**In plain English:**
1. Split input into two paths
2. Apply SiLU to first path
3. Keep second path as-is
4. Multiply them together element-wise

## How It Works

**Example:**

```python
import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W1 = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        # Path 1: Linear + SiLU
        gate = self.silu(self.W1(x))
        
        # Path 2: Linear only
        value = self.V(x)
        
        # Multiply together
        output = gate * value
        return output

# Test
swiglu = SwiGLU(dim=128)
x = torch.randn(32, 128)  # Batch of 32
output = swiglu(x)

print(output.shape)  # torch.Size([32, 128])
```

**Manual calculation (simplified):**

```yaml
Input x = [1.0, 2.0, 3.0]

Path 1 (Gate):
  W1(x) = [-0.5, 2.0, 1.0]
  SiLU(W1(x)) = [-0.19, 1.76, 0.73]

Path 2 (Value):
  V(x) = [0.8, -1.2, 2.0]

Element-wise multiply:
  [-0.19 * 0.8,  1.76 * -1.2,  0.73 * 2.0]
  = [-0.15, -2.11, 1.46]

The gate controls how much of value passes through!
```

## Why SwiGLU is Powerful

### 1. Gating Mechanism

```python
# Gating allows selective information flow
gate = torch.tensor([0.1, 0.5, 0.9])  # Low, medium, high gates
value = torch.tensor([5.0, 5.0, 5.0])  # Same values

output = gate * value
print(output)
# tensor([0.5, 2.5, 4.5])

# Gate controls how much gets through!
```

### 2. Double the Parameters (More Capacity)

```yaml
Regular FFN:
  Linear(dim, 4*dim) → ReLU → Linear(4*dim, dim)
  Parameters: dim*4*dim + 4*dim*dim = 8*dim²

SwiGLU:
  Two parallel linears + gating
  Parameters: Slightly more (~1.5x FFN)
  
But: Better performance despite similar size!
```

### 3. Smooth Activation (SiLU)

Using SiLU instead of ReLU provides smooth gradients!

## The GLU Family

![GLU Variants](/content/learn/activation-functions/swiglu/glu-variants.png)

All GLU variants follow the same pattern:

```yaml
GLU:     σ(W(x)) ⊙ V(x)    ← Sigmoid gate
ReGLU:   ReLU(W(x)) ⊙ V(x)  ← ReLU gate
GEGLU:   GELU(W(x)) ⊙ V(x)  ← GELU gate
SwiGLU:  SiLU(W(x)) ⊙ V(x)  ← SiLU gate (best!)
```

**Performance ranking (empirical):**

```yaml
Best:  SwiGLU ≈ GEGLU
Good:  ReGLU
Original: GLU
```

## Using SwiGLU in Transformers

SwiGLU is used in the feedforward network (FFN) of transformers:

```python
import torch
import torch.nn as nn

class SwiGLUFFN(nn.Module):
    """Feedforward network with SwiGLU"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * 8/3)  # Adjusted for gating
        
        self.W1 = nn.Linear(dim, hidden_dim, bias=False)
        self.V = nn.Linear(dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, dim, bias=False)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        # SwiGLU activation
        gate = self.silu(self.W1(x))
        value = self.V(x)
        hidden = gate * value
        
        # Project back
        output = self.W2(hidden)
        return output

# Example usage in transformer block
class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8)
        self.ffn = SwiGLUFFN(dim)  # ← SwiGLU FFN
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Attention block
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # FFN block with SwiGLU
        x = x + self.ffn(self.norm2(x))
        return x
```

## Where SwiGLU is Used

**Major models using SwiGLU:**
- **LLaMA** (Meta's language model)
- **PaLM** (Google's language model)
- **GPT-J** (EleutherAI)
- Many other modern LLMs

**Why they chose SwiGLU:**

```yaml
Research findings:
  - Better performance than standard FFN
  - Improved training stability
  - Smoother optimization
  - State-of-the-art results

Trade-off: Slightly more parameters, but worth it!
```

## Practical Example: LLaMA-style FFN

```python
import torch
import torch.nn as nn

class LLaMAFFN(nn.Module):
    """FFN from LLaMA (uses SwiGLU)"""
    def __init__(self, dim=4096, hidden_dim=11008):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)  # W1
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)    # V
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)  # W2
        self.silu = nn.SiLU()
    
    def forward(self, x):
        # SwiGLU
        gate = self.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        
        # Project back down
        output = self.down_proj(hidden)
        return output

# Test
ffn = LLaMAFFN(dim=512, hidden_dim=1376)  # Smaller for demo
x = torch.randn(2, 10, 512)  # Batch=2, seq_len=10, dim=512
output = ffn(x)

print(output.shape)  # torch.Size([2, 10, 512])
```

## Implementation Tips

### Efficient Implementation

```python
import torch
import torch.nn as nn

class EfficientSwiGLU(nn.Module):
    """Efficient SwiGLU with combined projection"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Combine W1 and V into single matrix for efficiency
        self.combined = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        # Single matrix multiply, then split
        combined = self.combined(x)
        gate, value = combined.chunk(2, dim=-1)
        
        # SwiGLU
        hidden = self.silu(gate) * value
        output = self.down(hidden)
        return output
```

## Key Takeaways

✓ **Gated activation:** One path controls another

✓ **Formula:** SiLU(W₁(x)) ⊙ V(x)

✓ **State-of-the-art:** Used in LLaMA, PaLM, and modern LLMs

✓ **Better than FFN:** Outperforms standard ReLU-based networks

✓ **Smooth:** Thanks to SiLU activation

✓ **More parameters:** But worth it for performance

**Quick Reference:**

```python
# Basic SwiGLU implementation
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(dim, hidden_dim)
        self.V = nn.Linear(dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        gate = torch.nn.functional.silu(self.W1(x))
        value = self.V(x)
        hidden = gate * value
        return self.W2(hidden)

# Usage
swiglu = SwiGLU(dim=512, hidden_dim=2048)
output = swiglu(input_tensor)
```

**When to use SwiGLU:**
- ✓ Transformer feedforward networks
- ✓ Large language models
- ✓ When you want state-of-the-art performance
- ✓ Modern architectures

**Remember:** SwiGLU is the advanced gating mechanism powering modern LLMs! 🎉
