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

## The Mathematical Foundation of Learning

Machine learning is fundamentally an **optimization problem**. We want to find parameters (weights and biases) that minimize a loss function.

**Mathematically:**
```
Find θ* = argmin L(θ)
          θ

Where:
  θ = model parameters (weights and biases)
  L(θ) = loss function
  θ* = optimal parameters that minimize loss
  argmin = "argument that minimizes"
```

### Why Gradient Descent?

For complex neural networks, we can't solve this analytically. Instead, we use **gradient descent**: an iterative algorithm that follows the slope downhill.

**Geometric intuition:**
- Loss function creates a "landscape" over parameter space
- We start at a random point (random weights)
- We look at the slope (gradient)
- We take a step downhill (opposite to gradient)
- Repeat until we reach a minimum!

## What Does "Learning" Mean?

**Learning = Automatically adjusting weights to make better predictions**

### The Calculus Behind Learning

**Gradient**: The vector of partial derivatives showing how loss changes with each parameter.

```
∇L(θ) = [∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ]

Where:
  ∇ (nabla) = gradient operator
  ∂L/∂θᵢ = partial derivative of loss w.r.t. parameter i
```

**Interpretation:**
- If ∂L/∂θᵢ > 0: increasing θᵢ increases loss → decrease θᵢ
- If ∂L/∂θᵢ < 0: increasing θᵢ decreases loss → increase θᵢ
- Magnitude |∂L/∂θᵢ| shows how sensitive loss is to θᵢ

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

**The algorithm that powers learning.**

### Mathematical Definition

**Update rule:**
```
θₜ₊₁ = θₜ - η ∇L(θₜ)

Where:
  θₜ = parameters at step t
  θₜ₊₁ = updated parameters
  η (eta) = learning rate (step size)
  ∇L(θₜ) = gradient of loss at current parameters
```

**Step-by-step:**
1. Compute loss: L(θₜ)
2. Compute gradient: ∇L(θₜ) = [∂L/∂θ₁, ∂L/∂θ₂, ...]
3. Update each parameter: θᵢ ← θᵢ - η(∂L/∂θᵢ)
4. Repeat!

### Example Calculation

Consider a single weight w:

```
Current state:
  w = 0.5
  L(w) = (ŷ - y)² where ŷ = wx
  
Suppose:
  x = 2.0, y = 3.0 (true value)
  ŷ = 0.5 × 2.0 = 1.0 (prediction)
  L = (1.0 - 3.0)² = 4.0 (high loss!)

Gradient calculation:
  ∂L/∂w = ∂/∂w[(wx - y)²]
        = 2(wx - y) · x         (chain rule)
        = 2(1.0 - 3.0) · 2.0
        = -8.0

Update (η = 0.1):
  w_new = w - η(∂L/∂w)
        = 0.5 - 0.1×(-8.0)
        = 0.5 + 0.8
        = 1.3

Check: New prediction = 1.3 × 2.0 = 2.6 (closer to 3.0!)
```

**Worked example:**

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

### The Mathematics of Convergence

**Why does gradient descent work?**

By Taylor expansion, for small η:
```
L(θ - η∇L) ≈ L(θ) - η||∇L||² + O(η²)
```

Since ||∇L||² > 0, if η is small enough:
```
L(θ - η∇L) < L(θ)
```

So each step reduces the loss! (Assuming η is not too large)

**Convergence conditions:**
- Loss is bounded below
- Gradients are Lipschitz continuous
- Learning rate satisfies: Σηₜ = ∞, Σηₜ² < ∞

## Learning Rate

**Learning rate controls how big each step is.**

### Mathematical Analysis

The learning rate η is a hyperparameter that balances:
- **Speed**: Larger η → faster convergence
- **Stability**: Smaller η → more stable, less oscillation

**Optimal learning rate theorem:**
For quadratic loss L(w) = ½w^T Aw:
```
Optimal η = 2/(λ_min + λ_max)

Where λ_min, λ_max are smallest/largest eigenvalues of A
```

In practice, we don't know this, so we:
- Start with η ≈ 0.01 or 0.001
- Use adaptive optimizers (Adam, RMSprop)
- Use learning rate schedules

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

### Mathematical Effects of Learning Rate

**Too small (η = 0.0001):**
```
Δθ = -η∇L = very small
→ Many iterations needed
→ May not reach minimum in reasonable time
```

**Just right (η = 0.01):**
```
Δθ = -η∇L = appropriate size
→ Steady progress
→ Converges efficiently
```

**Too large (η = 10):**
```
Δθ = -η∇L = too large
→ Overshoots minimum
→ Loss oscillates or diverges
```

## Simple Learning Example

Let's derive and implement gradient descent from first principles!

### Mathematical Setup

We want to learn the linear relationship y = wx + b:

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

### Detailed Gradient Derivation

Let's derive the gradients manually:

**Loss function:**
```
L = (1/n)Σᵢ(ŷᵢ - yᵢ)²
where ŷᵢ = wxᵢ + b
```

**Gradient w.r.t. w:**
```
∂L/∂w = ∂/∂w[(1/n)Σᵢ(wxᵢ + b - yᵢ)²]
      = (1/n)Σᵢ 2(wxᵢ + b - yᵢ) · xᵢ    (chain rule)
      = (2/n)Σᵢ(wxᵢ + b - yᵢ)xᵢ
```

**Gradient w.r.t. b:**
```
∂L/∂b = ∂/∂b[(1/n)Σᵢ(wxᵢ + b - yᵢ)²]
      = (1/n)Σᵢ 2(wxᵢ + b - yᵢ) · 1      (chain rule)
      = (2/n)Σᵢ(wxᵢ + b - yᵢ)
```

**Update rules:**
```
w ← w - η(∂L/∂w)
b ← b - η(∂L/∂b)
```

These are exactly what PyTorch computes with `loss.backward()`!

## What the Neuron Learns

### Feature Learning

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
