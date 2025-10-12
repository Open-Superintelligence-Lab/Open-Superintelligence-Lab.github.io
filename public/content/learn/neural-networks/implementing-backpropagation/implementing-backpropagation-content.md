---
hero:
  title: "Implementing Backpropagation"
  subtitle: "Coding the Backward Pass"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 10 min read"
---

Backpropagation is how neural networks **learn**. It calculates gradients for all weights efficiently!

## What is Backpropagation?

Imagine you're trying to improve a recipe. You taste the final dish (make a prediction), compare it to what you wanted (calculate loss), then figure out which ingredient to adjust (compute gradients). That's backpropagation!

**Mathematically:** Backpropagation uses the chain rule from calculus to compute how each weight contributes to the error. It works backwards from the output to the input, hence "back-propagation."

### Why It's Revolutionary

Before backpropagation (1986, Rumelhart et al.), training neural networks was impractical. Backpropagation made it possible to train networks with millions of parameters efficiently.

**The key insight:** By storing intermediate values during the forward pass, we can compute all gradients in one backward pass. This is much faster than computing each gradient independently!

## The Four-Step Algorithm

Let's break down backpropagation into digestible steps:

### Step 1: Forward Pass

First, run data through the network to make predictions:

```python
import torch
import torch.nn as nn

# Create a simple network
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Forward pass
x = torch.randn(32, 10)  # 32 samples, 10 features
predictions = model(x)    # Get predictions
```

During this step, PyTorch remembers all the operations in a "computation graph."

### Step 2: Compute Loss

Measure how wrong the predictions are:

```python
criterion = nn.MSELoss()
y = torch.randn(32, 1)  # True values
loss = criterion(predictions, y)

print(f"Loss: {loss.item():.4f}")
```

The loss is a single number representing overall error.

### Step 3: Backward Pass (The Magic!)

This is where backpropagation happens:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

optimizer.zero_grad()  # Clear previous gradients
loss.backward()         # Compute gradients (backprop!)
```

**What `loss.backward()` does:**
- Traverses the computation graph backwards
- Applies chain rule at each operation
- Stores gradient for each parameter
- All automatically!

### Step 4: Update Weights

Apply the computed gradients to improve the model:

```python
optimizer.step()  # Update all weights
```

This subtracts learning_rate × gradient from each weight.

### Complete Training Step

Here's everything together:

```python
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

# Run one training step
x = torch.randn(32, 10)
y = torch.randn(32, 1)
loss = train_step(x, y)
print(f"Loss: {loss:.4f}")
```

**The cycle:** Forward → Loss → Backward → Update → Repeat!

## Understanding Manual Backpropagation

To truly understand what's happening, let's see backpropagation "under the hood" with a tiny network:

### The Network Structure

Let's build a minimal network to see gradients flowing:

```
x → [×w1] → ReLU → [×w2] → y
```

### Step 1: Define Variables

```python
import torch

# Input and weights (all require gradients)
x = torch.tensor([2.0], requires_grad=True)
w1 = torch.tensor([0.5], requires_grad=True)
w2 = torch.tensor([1.5], requires_grad=True)
```

We set `requires_grad=True` to tell PyTorch "track operations on these for backprop."

### Step 2: Forward Pass

Watch the data transform at each step:

```python
# First computation: multiply by w1
z1 = w1 * x
print(f"After w1: {z1.item()}")  # 0.5 × 2.0 = 1.0

# Activation: ReLU
a1 = torch.relu(z1)
print(f"After ReLU: {a1.item()}")  # max(0, 1.0) = 1.0

# Second computation: multiply by w2
y = w2 * a1
print(f"Final output: {y.item()}")  # 1.5 × 1.0 = 1.5
```

**The complete forward pass:**
```
x=2.0 → ×0.5 → z1=1.0 → ReLU → a1=1.0 → ×1.5 → y=1.5
```

### Step 3: Calculate Loss

```python
target = torch.tensor([3.0])
loss = (y - target) ** 2

print(f"Loss: {loss.item()}")  # (1.5 - 3.0)² = 2.25
```

Our prediction (1.5) is far from target (3.0), so loss is high.

### Step 4: Backpropagation

Now the magic - one line computes all gradients:

```python
loss.backward()  # This does ALL the backpropagation!

print(f"Gradient of w1: {w1.grad.item():.4f}")
print(f"Gradient of w2: {w2.grad.item():.4f}")
print(f"Gradient of x: {x.grad.item():.4f}")
```

**What PyTorch computed:**
```
∂L/∂w2 = ∂L/∂y × ∂y/∂w2 = 2(y-target) × a1 = 2(1.5-3.0) × 1.0 = -3.0
∂L/∂w1 = ∂L/∂y × ∂y/∂a1 × ∂a1/∂z1 × ∂z1/∂w1 = -3.0 × 1.5 × 1 × 2.0 = -9.0
```

All from one `backward()` call!

### Step 5: Understanding the Gradients

```python
print(f"w1 gradient: {w1.grad.item():.4f}")  # -9.0
print(f"w2 gradient: {w2.grad.item():.4f}")  # -3.0
```

**What the negative gradients tell us:**
- Negative gradient means: "Increase this weight to reduce loss"
- w1 should increase (currently 0.5, should be higher)
- w2 should increase (currently 1.5, should be higher)
- This makes sense: we need a larger output (1.5 → 3.0)

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
