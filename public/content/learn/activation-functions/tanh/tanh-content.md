---
hero:
  title: "Tanh"
  subtitle: "Hyperbolic Tangent - Zero-centered Activation"
  tags:
    - "⚡ Activation Functions"
    - "⏱️ 10 min read"
---

Tanh (hyperbolic tangent) is like Sigmoid's **zero-centered cousin**. It squashes inputs to the range **[-1, 1]** instead of [0, 1].

## The Formula

**tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)**

Or equivalently: **tanh(x) = 2·σ(2x) - 1** (scaled and shifted sigmoid)

![Tanh Graph](/content/learn/activation-functions/tanh/tanh-graph.png)

```yaml
Input → -∞  →  Output → -1
Input = 0   →  Output = 0
Input → +∞  →  Output → +1

Key property: Output is always in (-1, 1)
Zero-centered! (unlike sigmoid)
```

## How It Works

**Example:**

```python
import torch
import torch.nn as nn

# Create tanh activation
tanh = nn.Tanh()

# Test with different values
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tanh(x)

print(output)
# tensor([-0.9640, -0.7616,  0.0000,  0.7616,  0.9640])
```

**Manual calculation:**

```yaml
Input:  [-2.0, -1.0,  0.0,  1.0,  2.0]
         ↓      ↓      ↓     ↓     ↓
Tanh:   -0.96  -0.76  0.00  0.76  0.96
         ↓      ↓      ↓     ↓     ↓
Range:  All values between -1 and 1
```

## The Zero-Centered Advantage

**This is tanh's superpower:** outputs are centered around zero!

![Tanh vs Sigmoid](/content/learn/activation-functions/tanh/tanh-vs-sigmoid.png)

**Example:**

```python
import torch

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Tanh: zero-centered
tanh_out = torch.tanh(x)
print(tanh_out.mean())
# tensor(0.0000) ← Mean is zero!

# Sigmoid: NOT zero-centered
sigmoid_out = torch.sigmoid(x)
print(sigmoid_out.mean())
# tensor(0.5000) ← Mean is 0.5
```

**Why zero-centered is better:**

```yaml
Zero-centered (tanh):
  ✓ Gradients can be positive or negative
  ✓ Faster convergence
  ✓ More stable training
  ✓ Better for hidden layers

Not zero-centered (sigmoid):
  ✗ All gradients have same sign
  ✗ Slower learning
  ✗ Zig-zag optimization path
```

## In Code (Simple Implementation)

```python
import torch

def tanh_manual(x):
    """Manual tanh implementation"""
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

# Test it
x = torch.tensor([-1.0, 0.0, 1.0])
output = tanh_manual(x)
print(output)
# tensor([-0.7616,  0.0000,  0.7616])

# Verify against PyTorch
print(torch.tanh(x))
# tensor([-0.7616,  0.0000,  0.7616]) ← Same!
```

## Using Tanh in PyTorch

### Method 1: As a Layer

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.Tanh(),      # ← Tanh activation
    nn.Linear(20, 5),
    nn.Tanh(),      # ← Another tanh
    nn.Linear(5, 1)
)
```

### Method 2: As a Function

```python
import torch
import torch.nn.functional as F

x = torch.randn(5, 10)
output = F.tanh(x)  # or torch.tanh(x)
```

## Practical Example: RNN/LSTM

Tanh is commonly used in recurrent neural networks:

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, hidden):
        # Combine input and hidden state
        combined = self.i2h(x) + self.h2h(hidden)
        
        # Apply tanh
        new_hidden = torch.tanh(combined)  # ← Tanh here!
        return new_hidden

# Initialize
rnn = SimpleRNN(input_size=10, hidden_size=20)
x = torch.randn(5, 10)  # 5 samples
h = torch.zeros(5, 20)  # Initial hidden state

# Forward pass
new_h = rnn(x, h)
print(new_h.shape)  # torch.Size([5, 20])
print(new_h.min(), new_h.max())
# All values between -1 and 1!
```

## Tanh vs Sigmoid vs ReLU

```yaml
Tanh:
  ✓ Zero-centered (best for hidden layers)
  ✓ Output range: [-1, 1]
  ✓ Smooth gradient
  ✗ Vanishing gradient problem
  ✗ Slower than ReLU (exponentials)
  
Sigmoid:
  ✓ Output range: [0, 1] (probabilities)
  ✓ Smooth gradient
  ✗ NOT zero-centered
  ✗ Vanishing gradient problem
  ✗ Slower than ReLU
  
ReLU:
  ✓ Fast (no exponentials)
  ✓ No vanishing gradient for x > 0
  ✓ Creates sparsity
  ✗ NOT smooth at zero
  ✗ Dying ReLU problem
  ✗ NOT zero-centered
```

**When to use each:**

```yaml
Hidden layers:
  Modern: ReLU (fastest, works well)
  Classical: Tanh (zero-centered)
  Rarely: Sigmoid (not zero-centered)

Output layer:
  Binary classification: Sigmoid
  Multi-class: Softmax
  Regression: None (linear)
  
RNN/LSTM:
  Gates: Sigmoid
  State update: Tanh
```

## The Vanishing Gradient Problem

Like sigmoid, tanh suffers from vanishing gradients:

```python
import torch

# Large input
x = torch.tensor([5.0], requires_grad=True)
y = torch.tanh(x)
y.backward()

print(f"Output: {y.item():.6f}")  # 0.999909
print(f"Gradient: {x.grad.item():.6f}")  # 0.000181
# Gradient is tiny!
```

**Why this happens:**

```yaml
For large |x|:
  Output saturates (near -1 or +1)
  Gradient becomes very small
  Learning slows down
  
This is why ReLU replaced tanh in most modern networks!
```

## Relationship to Sigmoid

Tanh is actually just a rescaled sigmoid:

```python
import torch

x = torch.tensor([0.5, 1.0, 1.5])

# Tanh
tanh_output = torch.tanh(x)

# Same as scaled sigmoid
sigmoid_output = 2 * torch.sigmoid(2*x) - 1

print(tanh_output)
# tensor([0.4621, 0.7616, 0.9051])

print(sigmoid_output)
# tensor([0.4621, 0.7616, 0.9051])

# They're the same!
```

**Mathematical relationship:**

```yaml
tanh(x) = 2·sigmoid(2x) - 1

Proof:
  sigmoid(x) gives [0, 1]
  2·sigmoid(2x) gives [0, 2]
  2·sigmoid(2x) - 1 gives [-1, 1] ← tanh range!
```

## Key Takeaways

✓ **S-shaped curve:** Like sigmoid but zero-centered

✓ **Output range:** Always between -1 and 1

✓ **Zero-centered:** Better than sigmoid for hidden layers

✓ **Formula:** (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)

✓ **Common in RNNs:** Used in LSTM/GRU cells

✓ **Vanishing gradients:** Mostly replaced by ReLU in modern networks

**Quick Reference:**

```python
# Using tanh
import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Module
tanh_layer = nn.Tanh()
output = tanh_layer(x)

# Method 2: Functional
output = F.tanh(x)

# Method 3: Direct
output = torch.tanh(x)

# Method 4: Manual
output = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
```

**Remember:** Tanh is zero-centered sigmoid. Use it for RNN states, but ReLU is faster for feedforward! 🎉
