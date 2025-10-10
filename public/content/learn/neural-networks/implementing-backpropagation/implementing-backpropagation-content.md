---
hero:
  title: "Implementing Backpropagation"
  subtitle: "Coding the Backward Pass"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 10 min read"
---

Backpropagation is how neural networks **learn**. It calculates gradients for all weights efficiently!

## The Algorithm

**Backpropagation:**
1. Forward pass: Compute predictions
2. Compute loss
3. Backward pass: Compute gradients (chain rule)
4. Update weights

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training step
def train_step(x, y):
    # 1. Forward pass
    predictions = model(x)
    
    # 2. Compute loss
    loss = criterion(predictions, y)
    
    # 3. Backward pass (backpropagation!)
    optimizer.zero_grad()
    loss.backward()
    
    # 4. Update weights
    optimizer.step()
    
    return loss.item()

# Train
x = torch.randn(32, 10)
y = torch.randn(32, 1)
loss = train_step(x, y)
print(f"Loss: {loss:.4f}")
```

## Manual Backpropagation

```python
import torch

# Simple network: y = w2 * relu(w1 * x)
x = torch.tensor([2.0], requires_grad=True)
w1 = torch.tensor([0.5], requires_grad=True)
w2 = torch.tensor([1.5], requires_grad=True)

# Forward
z1 = w1 * x
a1 = torch.relu(z1)
y = w2 * a1

# Target
target = torch.tensor([3.0])
loss = (y - target) ** 2

# Backward (automatic)
loss.backward()

print(f"dL/dw1: {w1.grad.item()}")
print(f"dL/dw2: {a1.item()}")
```

## Key Takeaways

✓ **Backprop:** Efficiently computes all gradients

✓ **Chain rule:** Applied automatically by PyTorch

✓ **Three steps:** forward → backward → update

✓ **`.backward()`:** Does all the work!

**Quick Reference:**

```python
# Standard training step
optimizer.zero_grad()  # Clear old gradients
loss.backward()         # Compute gradients
optimizer.step()        # Update weights
```

**Remember:** Backpropagation = automatic gradient calculation! 🎉
