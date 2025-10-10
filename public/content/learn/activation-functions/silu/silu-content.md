---
hero:
  title: "SiLU"
  subtitle: "Sigmoid Linear Unit - The Swish Activation"
  tags:
    - "⚡ Activation Functions"
    - "⏱️ 10 min read"
---

SiLU (also called Swish) is a **smooth** alternative to ReLU. It's ReLU but with a smooth curve instead of a hard cutoff!

## The Formula

**SiLU(x) = x · σ(x) = x · sigmoid(x)**

Simply multiply the input by its sigmoid! This creates a smooth, non-linear function.

![SiLU Graph](/content/learn/activation-functions/silu/silu-graph.png)

```yaml
For large negative x:
  sigmoid(x) ≈ 0
  SiLU(x) = x · 0 ≈ 0

For x = 0:
  sigmoid(0) = 0.5
  SiLU(0) = 0 · 0.5 = 0

For large positive x:
  sigmoid(x) ≈ 1
  SiLU(x) = x · 1 ≈ x
```

## How It Works

**Example:**

```python
import torch
import torch.nn as nn

# Create SiLU activation
silu = nn.SiLU()

# Test with different values
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = silu(x)

print(output)
# tensor([-0.2384, -0.2689,  0.0000,  0.7311,  1.7616])
```

**Manual calculation (for x = 2):**

```yaml
SiLU(2) = 2 · sigmoid(2)
        = 2 · (1 / (1 + e⁻²))
        = 2 · 0.881
        = 1.762

Notice: Not just 2 (like ReLU), but close!
```

## The Smooth Advantage

Unlike ReLU, SiLU is **smooth everywhere** and allows small negative values:

![SiLU vs ReLU](/content/learn/activation-functions/silu/silu-vs-relu.png)

**Example comparison:**

```python
import torch

x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 1.0, 2.0])

# ReLU: hard cutoff
relu_out = torch.relu(x)
print("ReLU:", relu_out)
# tensor([0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 2.0000])

# SiLU: smooth transition
silu_out = torch.nn.functional.silu(x)
print("SiLU:", silu_out)
# tensor([-0.2384, -0.2689, -0.1887,  0.0000,  0.7311,  1.7616])
```

**Key differences:**

```yaml
ReLU:
  x < 0 → Output = 0 (hard cutoff)
  x > 0 → Output = x (straight line)
  NOT smooth at x = 0

SiLU:
  x < 0 → Small negative values (smooth)
  x > 0 → Nearly linear (smooth)
  Smooth everywhere!
```

## Why SiLU is Better Than ReLU

### 1. Smooth Gradients

```python
import torch

x = torch.tensor([0.0], requires_grad=True)

# ReLU gradient at x=0 is undefined (jump)
# SiLU gradient at x=0 is smooth (0.5)
y = torch.nn.functional.silu(x)
y.backward()

print(x.grad)  # tensor([0.5000])
# Smooth gradient!
```

### 2. No Dying Neurons

```python
# Neuron that would "die" with ReLU
x = torch.tensor([-5.0], requires_grad=True)

# ReLU would output 0 with gradient 0
relu_out = torch.relu(x)
print(relu_out)  # tensor([0.]) ← Dead!

# SiLU allows gradient flow
silu_out = torch.nn.functional.silu(x)
print(silu_out)  # tensor([-0.0337]) ← Small but not zero!

# Gradient still flows
silu_out.backward()
print(x.grad)  # tensor([0.0030]) ← Can still learn!
```

### 3. Better Performance

Recent research shows SiLU **outperforms ReLU** in many tasks, especially in vision transformers and modern architectures!

## In Code (Simple Implementation)

```python
import torch

def silu(x):
    """Simple SiLU implementation"""
    return x * torch.sigmoid(x)

# Test it
x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
output = silu(x)
print(output)
# tensor([-0.2689,  0.0000,  0.7311,  1.7616])

# Verify against PyTorch
print(torch.nn.functional.silu(x))
# tensor([-0.2689,  0.0000,  0.7311,  1.7616]) ← Same!
```

## Using SiLU in PyTorch

### Method 1: As a Layer

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.SiLU(),       # ← SiLU activation
    nn.Linear(20, 5),
    nn.SiLU(),       # ← Another SiLU
    nn.Linear(5, 1)
)
```

### Method 2: As a Function

```python
import torch.nn.functional as F

x = torch.randn(5, 10)
output = F.silu(x)
```

## Practical Example: Vision Transformer

SiLU is used in many modern architectures like EfficientNet and Vision Transformers:

```python
import torch
import torch.nn as nn

class ModernBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.silu = nn.SiLU()  # ← SiLU instead of ReLU!
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.silu(x)  # Smooth activation
        x = self.fc2(x)
        return x + residual

# Test
block = ModernBlock(dim=128)
x = torch.randn(32, 128)  # Batch of 32
output = block(x)
print(output.shape)  # torch.Size([32, 128])
```

## SiLU vs Other Activations

```yaml
SiLU (Swish):
  ✓ Smooth everywhere (no hard cutoff)
  ✓ No dying neurons
  ✓ Better performance than ReLU
  ✓ Self-gated (uses its own sigmoid)
  ✗ Slightly slower than ReLU
  ✗ More computation (sigmoid)

ReLU:
  ✓ Fastest (simple comparison)
  ✓ Simple to understand
  ✗ Not smooth at x=0
  ✗ Dying neuron problem
  ✗ Hard cutoff at zero

Tanh:
  ✓ Zero-centered
  ✓ Smooth
  ✗ Vanishing gradients
  ✗ Slower than both
```

## Where SiLU is Used

**Modern architectures using SiLU:**
- EfficientNet (image classification)
- Vision Transformers (ViT)
- Some language models
- Mobile-optimized networks

**Example from research:**

```yaml
Study: "Searching for Activation Functions" (Google Brain, 2017)
Finding: Swish/SiLU outperformed ReLU on ImageNet
Result: Adopted in many modern architectures

Performance gain: ~0.6-0.9% accuracy improvement
```

## Practical Example: EfficientNet-style Block

```python
import torch
import torch.nn as nn

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck with SiLU"""
    def __init__(self, in_channels, out_channels, expand_ratio=4):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 1)
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, 
                                        padding=1, groups=hidden_dim)
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, 1)
        self.silu = nn.SiLU()  # ← SiLU for smooth activation
    
    def forward(self, x):
        # Expand
        out = self.expand_conv(x)
        out = self.silu(out)  # SiLU
        
        # Depthwise
        out = self.depthwise_conv(out)
        out = self.silu(out)  # SiLU
        
        # Project
        out = self.project_conv(out)
        return out

# Test
block = MBConvBlock(32, 64)
x = torch.randn(1, 32, 56, 56)  # Image: batch, channels, H, W
output = block(x)
print(output.shape)  # torch.Size([1, 64, 56, 56])
```

## The Self-Gating Mechanism

SiLU is "self-gated" - it uses its own sigmoid as a gate:

```python
import torch

x = torch.tensor([2.0])

# SiLU gates itself
sigmoid_gate = torch.sigmoid(x)  # 0.881
output = x * sigmoid_gate         # 2.0 * 0.881 = 1.762

print(f"Input: {x.item()}")
print(f"Gate: {sigmoid_gate.item():.3f}")
print(f"Output: {output.item():.3f}")

# Input: 2.0
# Gate: 0.881
# Output: 1.762
```

**What this means:**

```yaml
The input controls its own "gate":
  - Large positive x → gate ≈ 1 → mostly pass through
  - Large negative x → gate ≈ 0 → mostly blocked
  - Small x → partial gating (smooth)

This self-regulation makes SiLU effective!
```

## Key Takeaways

✓ **Formula:** x · sigmoid(x)

✓ **Smooth:** No hard cutoff like ReLU

✓ **Self-gated:** Uses its own sigmoid as a gate

✓ **Better than ReLU:** Improved performance in many tasks

✓ **No dying neurons:** Always has gradient flow

✓ **Modern choice:** Used in EfficientNet, ViT, and more

**Quick Reference:**

```python
# Using SiLU
import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Module
silu_layer = nn.SiLU()
output = silu_layer(x)

# Method 2: Functional
output = F.silu(x)

# Method 3: Manual
output = x * torch.sigmoid(x)

# Also known as Swish
swish = nn.SiLU()  # Same thing!
```

**When to use SiLU:**
- ✓ Modern CNN architectures
- ✓ Vision transformers
- ✓ When you want better performance than ReLU
- ✓ Mobile/efficient networks

**Remember:** SiLU is the smooth, modern upgrade to ReLU! 🎉
