---
hero:
  title: "The Chain Rule"
  subtitle: "The Math Behind Backpropagation"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 8 min read"
---

The chain rule is how we calculate gradients through multiple layers. It's the secret sauce of backpropagation!

## The Basic Idea

**Chain rule: Multiply gradients as you go backwards through layers**

```yaml
If y = f(g(x)), then:
dy/dx = (dy/dg) × (dg/dx)

In words: Multiply the gradients of each function
```

## Simple Example

```python
import torch

# y = (x + 2)²
x = torch.tensor([3.0], requires_grad=True)

# Break it down:
# g = x + 2
# y = g²

g = x + 2
y = g ** 2

# Backward pass
y.backward()

print(f"x = {x.item()}")
print(f"g = {g.item()}")
print(f"y = {y.item()}")
print(f"dy/dx = {x.grad.item()}")

# Manual:
# dy/dg = 2g = 2×5 = 10
# dg/dx = 1
# dy/dx = 10×1 = 10 ✓
```

## In Neural Networks

```python
import torch
import torch.nn as nn

# Two-layer network
model = nn.Sequential(
    nn.Linear(1, 1),  # Layer 1
    nn.ReLU(),
    nn.Linear(1, 1)   # Layer 2
)

x = torch.tensor([[2.0]])
y_true = torch.tensor([[10.0]])

# Forward
y_pred = model(x)
loss = (y_pred - y_true) ** 2

# Backward (chain rule applied automatically!)
loss.backward()

# Gradients computed through both layers
for name, param in model.named_parameters():
    print(f"{name}: gradient = {param.grad}")
```

**What happens:**

```yaml
Forward:
  x → Layer1 → ReLU → Layer2 → prediction → loss

Backward (chain rule):
  dloss/dprediction → dLayer2 → dReLU → dLayer1 → dx
  
Each gradient multiplies with the next!
```

## Why It Works

```yaml
Loss depends on layer 2 output
Layer 2 output depends on ReLU output  
ReLU output depends on layer 1 output
Layer 1 output depends on weights

So: Loss depends on weights (through chain)!

Chain rule connects them:
dLoss/dWeight = dLoss/dOutput × dOutput/dWeight
```

## PyTorch Does It For You

```python
import torch

# Complex computation
x = torch.tensor([2.0], requires_grad=True)
y = ((x ** 2 + 3) * torch.sin(x)) ** 3

# PyTorch applies chain rule automatically!
y.backward()

print(f"Gradient: {x.grad.item()}")
# Calculated using chain rule through all operations!
```

## Key Takeaways

✓ **Chain rule:** Multiply gradients backwards

✓ **Backpropagation:** Applies chain rule through network

✓ **Automatic:** PyTorch does it for you

✓ **Essential:** Makes training deep networks possible

**Remember:** Chain rule lets us train deep networks by connecting all the gradients! 🎉
