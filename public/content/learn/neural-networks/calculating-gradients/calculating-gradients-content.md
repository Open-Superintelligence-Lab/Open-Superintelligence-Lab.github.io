---
hero:
  title: "Calculating Gradients"
  subtitle: "Understanding Gradient Computation"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 8 min read"
---

Gradients tell us **which direction** to adjust weights to reduce loss!

## What is a Gradient?

**Gradient = Rate of change of loss with respect to a parameter**

```python
import torch

# Simple function: loss = w²
w = torch.tensor([3.0], requires_grad=True)
loss = w ** 2

# Calculate gradient
loss.backward()

print(f"Weight: {w.item()}")
print(f"Loss: {loss.item()}")
print(f"Gradient: {w.grad.item()}")

# Gradient = 2w = 2×3 = 6
# This tells us: increasing w increases loss
```

## Computing Gradients in PyTorch

```python
import torch
import torch.nn as nn

# Model
model = nn.Linear(3, 1)

# Data
x = torch.tensor([[1.0, 2.0, 3.0]])
y_true = torch.tensor([[5.0]])

# Forward pass
y_pred = model(x)
loss = (y_pred - y_true) ** 2

# Compute gradients
loss.backward()

# Check gradients
print("Weight gradients:", model.weight.grad)
print("Bias gradient:", model.bias.grad)
```

## Gradient Descent Update

```python
# Manual gradient descent
learning_rate = 0.01

with torch.no_grad():
    for param in model.parameters():
        # Update: param = param - lr * gradient
        param -= learning_rate * param.grad
        
        # Reset gradient
        param.grad.zero_()
```

## Key Takeaways

✓ **Gradient:** Direction and magnitude of change

✓ **`.backward()`:** Computes all gradients

✓ **Automatic:** PyTorch calculates for you

✓ **Update rule:** param -= lr * gradient

**Quick Reference:**

```python
# Compute gradients
loss.backward()

# Access gradients
param.grad

# Zero gradients
optimizer.zero_grad()
# or
param.grad.zero_()
```

**Remember:** Gradients point the way to better weights! 🎉
