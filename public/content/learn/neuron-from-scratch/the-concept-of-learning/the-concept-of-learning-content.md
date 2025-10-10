---
hero:
  title: "The Concept of Learning"
  subtitle: "How Neurons Adjust Their Weights"
  tags:
    - "🧠 Neuron"
    - "⏱️ 8 min read"
---

Learning is the process of **adjusting weights to reduce loss**. The neuron literally learns from mistakes!

![Learning Process](/content/learn/neuron-from-scratch/the-concept-of-learning/learning-process.png)

## What Does "Learning" Mean?

**Learning = Automatically adjusting weights to make better predictions**

```yaml
Before learning:
  Weights: Random
  Predictions: Bad
  Loss: High

After learning:
  Weights: Optimized
  Predictions: Good
  Loss: Low
```

## The Learning Process

**Step-by-step:**

1. Make prediction (forward pass)
2. Calculate loss (how wrong?)
3. Calculate gradients (which direction to adjust?)
4. Update weights (move in right direction)
5. Repeat!

**Example:**

```python
import torch
import torch.nn as nn

# Model
model = nn.Linear(1, 1)

# Training data
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])  # y = 2x

# Loss function
criterion = nn.MSELoss()

# Optimizer (handles weight updates)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # 1. Forward pass
    predictions = model(x)
    
    # 2. Calculate loss
    loss = criterion(predictions, y)
    
    # 3. Backward pass (calculate gradients)
    optimizer.zero_grad()
    loss.backward()
    
    # 4. Update weights
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# After training
print(f"Learned weight: {model.weight.item():.2f}")  # Should be close to 2.0
print(f"Learned bias: {model.bias.item():.2f}")      # Should be close to 0.0
```

## Gradient Descent

**The algorithm that powers learning:**

```yaml
Current weight: w = 0.5
Loss: high

Gradient: ∂Loss/∂w = -2.3
  Negative gradient → loss decreases if we INCREASE w

Update:
  w_new = w - learning_rate × gradient
  w_new = 0.5 - 0.01 × (-2.3)
  w_new = 0.5 + 0.023
  w_new = 0.523

Result: Loss is now lower!
```

## Learning Rate

**Learning rate controls how big each step is:**

```python
# Too small: slow learning
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
# Takes forever to learn!

# Just right: good learning
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Learns efficiently

# Too large: unstable learning
optimizer = torch.optim.SGD(model.parameters(), lr=10.0)
# Might overshoot and never converge!
```

**Effect of learning rate:**

```yaml
lr = 0.001 (small):
  Small weight updates
  Slow but stable
  Many epochs needed

lr = 0.01 (medium):
  Moderate updates
  Good balance
  Converges reasonably

lr = 1.0 (large):
  Large weight updates
  Fast but unstable
  Might oscillate or diverge
```

## Simple Learning Example

```python
import torch

# True relationship: y = 3x + 1
x_train = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_train = torch.tensor([4.0, 7.0, 10.0, 13.0])

# Model (start with random weights)
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

learning_rate = 0.01

# Train for 100 steps
for step in range(100):
    # Prediction
    y_pred = w * x_train + b
    
    # Loss
    loss = ((y_pred - y_train) ** 2).mean()
    
    # Backpropagation
    loss.backward()
    
    # Update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Reset gradients
        w.grad.zero_()
        b.grad.zero_()
    
    if step % 20 == 0:
        print(f"Step {step}: w={w.item():.2f}, b={b.item():.2f}, loss={loss.item():.4f}")

print(f"\\nLearned: y = {w.item():.2f}x + {b.item():.2f}")
# Should be close to: y = 3x + 1
```

## What the Neuron Learns

```python
# Example: Learning to classify

# Initially (random weights):
prediction = neuron([1.0, 2.0])  # 0.34 (wrong!)
actual = 1.0
loss = high

# After seeing examples:
# The neuron learns that:
# - Feature 1 with value > 0.5 → usually class 1
# - Feature 2 with value > 1.0 → usually class 1
# So it adjusts weights accordingly

# Finally (trained weights):
prediction = neuron([1.0, 2.0])  # 0.98 (correct!)
actual = 1.0
loss = low
```

## Key Takeaways

✓ **Learning = Adjusting weights:** Based on errors

✓ **Goal:** Minimize loss

✓ **Gradient descent:** The learning algorithm

✓ **Learning rate:** Controls step size

✓ **Automatic:** PyTorch calculates gradients for you!

**Quick Reference:**

```python
# Training loop
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(inputs)
    
    # Calculate loss
    loss = criterion(predictions, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
```

**Remember:** Learning is just: predict → measure error → adjust → repeat! 🎉
