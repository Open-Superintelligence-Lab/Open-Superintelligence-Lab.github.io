---
hero:
  title: "Calculating Gradients"
  subtitle: "Understanding Gradient Computation"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 8 min read"
---

Gradients tell us **which direction** to adjust weights to reduce loss!

## The Mathematical Essence

Gradients are the foundation of how neural networks learn. Without understanding gradients, deep learning remains a black box. Let's demystify them!

### What is a Gradient?

**Gradient = Rate of change of loss with respect to a parameter**

Mathematically:
```
∂L/∂w = lim[h→0] (L(w+h) - L(w))/h

This tells us: "If I change w by a tiny amount, how does L change?"
```

**In practical terms:**
- Positive gradient: Increasing weight increases loss → decrease weight
- Negative gradient: Increasing weight decreases loss → increase weight
- Large |gradient|: Loss is very sensitive to this weight
- Small |gradient|: Loss barely affected by this weight

### A Simple Example

Let's compute a gradient manually, then verify with PyTorch:

**Function:** loss = w²

**Manual derivative:**
```
d/dw (w²) = 2w
```

If w = 3, then gradient = 2(3) = 6

**In code:**

```python
import torch

# Simple function: loss = w²
w = torch.tensor([3.0], requires_grad=True)
loss = w ** 2

# Calculate gradient
loss.backward()

print(f"Weight: {w.item()}")        # 3.0
print(f"Loss: {loss.item()}")       # 9.0
print(f"Gradient: {w.grad.item()}")  # 6.0
```

**Interpretation:**
```
Gradient = 6.0 (positive)
→ If we increase w, loss increases
→ If we decrease w, loss decreases
→ So we should decrease w to minimize loss!
```

Let's verify:
```
Current: w=3, loss=9
If w=2.9: loss=8.41 (decreased!) ✓
If w=3.1: loss=9.61 (increased!) ✓
```

## Computing Gradients in Neural Networks

Now let's see gradients in a real network with multiple parameters:

### Step 1: Create a Simple Network

```python
import torch
import torch.nn as nn

# Single neuron: 3 inputs → 1 output
model = nn.Linear(3, 1)

# Check initial parameters
print(f"Weights shape: {model.weight.shape}")  # torch.Size([1, 3])
print(f"Bias shape: {model.bias.shape}")        # torch.Size([1])
```

**Parameters:** 3 weights + 1 bias = 4 learnable parameters

### Step 2: Prepare Data

```python
x = torch.tensor([[1.0, 2.0, 3.0]])  # Input
y_true = torch.tensor([[5.0]])        # Target
```

### Step 3: Forward Pass

```python
y_pred = model(x)  # Make prediction
print(f"Prediction: {y_pred.item():.3f}")

loss = (y_pred - y_true) ** 2  # Calculate loss
print(f"Loss: {loss.item():.3f}")
```

Let's say the prediction is 2.5. Then:
```
Loss = (2.5 - 5.0)² = (-2.5)² = 6.25
```

Our prediction is way off!

### Step 4: Compute Gradients

This is where the magic happens:

```python
loss.backward()  # Compute gradients for ALL parameters

# Check gradients for each parameter
print("Weight gradients:", model.weight.grad)
# tensor([[-5.0000, -10.0000, -15.0000]])

print("Bias gradient:", model.bias.grad)
# tensor([-5.0000])
```

**What these gradients mean:**
```
∂L/∂w₁ = -5.0  → Increasing w₁ decreases loss (negative gradient)
∂L/∂w₂ = -10.0 → Increasing w₂ decreases loss even more!
∂L/∂w₃ = -15.0 → Increasing w₃ has the strongest effect
∂L/∂b = -5.0   → Increasing bias decreases loss
```

All gradients are negative, telling us to **increase all parameters** to reduce loss.

### Why Different Gradient Magnitudes?

Notice w₃ has the largest gradient (-15). Why?

```
∂L/∂wᵢ depends on the input xᵢ!

Input was [1.0, 2.0, 3.0]
  x₃ = 3.0 is largest
  → w₃ has largest impact on output
  → w₃ has largest gradient

This is the chain rule at work!
```

## Using Gradients to Update Weights

Now that we have gradients, let's use them to improve our model:

### The Update Rule

Gradient descent update: **new_param = old_param - learning_rate × gradient**

```python
learning_rate = 0.01

# Manual update (PyTorch normally does this for you)
with torch.no_grad():  # Don't track these operations
    for param in model.parameters():
        # The update step
        param -= learning_rate * param.grad
        
        # Reset gradient for next iteration
        param.grad.zero_()
```

### Understanding the Update

Let's trace what happens to one weight:

```
Before:
  w₁ = 0.5
  gradient = -5.0
  learning_rate = 0.01

Update:
  w₁_new = w₁ - (learning_rate × gradient)
        = 0.5 - (0.01 × -5.0)
        = 0.5 - (-0.05)
        = 0.5 + 0.05
        = 0.55

After: w₁ = 0.55 (increased!)
```

**Why did it increase?** Negative gradient means "go up" to reduce loss!

### The Complete Training Cycle

```python
# Full cycle with visualization
print("Before training:")
print(f"  Prediction: {model(x).item():.3f}")
print(f"  Loss: {loss.item():.3f}")

# One update step
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Check improvement
new_pred = model(x)
new_loss = (new_pred - y_true) ** 2

print("\\nAfter one update:")
print(f"  Prediction: {new_pred.item():.3f}")
print(f"  Loss: {new_loss.item():.3f}")
print(f"  Improvement: {(loss.item() - new_loss.item()):.3f}")
```

**Expected output:**
```
Before training:
  Prediction: 2.500
  Loss: 6.250

After one update:
  Prediction: 2.650
  Loss: 5.523
  Improvement: 0.727
```

One step closer to the target!

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
