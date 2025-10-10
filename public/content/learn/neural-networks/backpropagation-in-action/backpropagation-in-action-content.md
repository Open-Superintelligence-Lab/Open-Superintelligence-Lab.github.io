---
hero:
  title: "Backpropagation in Action"
  subtitle: "Seeing Gradients Flow Through Networks"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 8 min read"
---

Let's see backpropagation in action with real examples!

## Watching Gradients

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

x = torch.tensor([[1.0, 2.0]])
y_true = torch.tensor([[5.0]])

# Forward
y_pred = model(x)
loss = (y_pred - y_true) ** 2

# Backward
loss.backward()

# See gradients
for name, param in model.named_parameters():
    print(f"{name}:")
    print(f"  Value: {param.data}")
    print(f"  Gradient: {param.grad}")
    print()
```

## Gradient Flow Example

```python
import torch

# Three-step computation
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2      # y = x²
z = y + 3       # z = y + 3  
loss = z ** 2   # loss = z²

# Backward
loss.backward()

print(f"x = {x.item()}")
print(f"y = {y.item()}")
print(f"z = {z.item()}")
print(f"loss = {loss.item()}")
print(f"\\ndloss/dx = {x.grad.item()}")

# Manual chain rule:
# dloss/dx = dloss/dz × dz/dy × dy/dx
#          = 2z × 1 × 2x
#          = 2(7) × 1 × 2(2)
#          = 14 × 4 = 56 ✓
```

## Training with Backprop

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Data: y = 2x
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Train
for epoch in range(50):
    # Forward
    pred = model(X)
    loss = criterion(pred, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Update
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print(f"Learned weight: {model.weight.item():.2f}")  # ~2.0
print(f"Learned bias: {model.bias.item():.2f}")      # ~0.0
```

## Key Takeaways

✓ **Backprop:** Computes gradients efficiently

✓ **Chain rule:** Multiplies gradients backwards

✓ **Automatic:** PyTorch handles it

✓ **Essential:** Makes training possible

**Remember:** Backprop = automatic gradient calculation through layers! 🎉
