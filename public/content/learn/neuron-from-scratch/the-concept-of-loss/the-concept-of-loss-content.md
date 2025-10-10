---
hero:
  title: "The Concept of Loss"
  subtitle: "Measuring How Wrong Your Model Is"
  tags:
    - "🧠 Neuron"
    - "⏱️ 8 min read"
---

Loss tells you **how wrong** your model's predictions are. Lower loss = better model!

![Loss Function](/content/learn/neuron-from-scratch/the-concept-of-loss/loss-function.png)

## What is Loss?

**Loss = Difference between prediction and actual answer**

Think of it like a score in golf - **lower is better**!

**Example:**

```python
import torch

# Actual answer (ground truth)
actual = torch.tensor([1.0])

# Model's prediction
prediction = torch.tensor([0.7])

# Loss: how far off?
loss = (prediction - actual) ** 2  # Squared difference
print(loss)
# tensor([0.0900])

# Closer prediction
better_prediction = torch.tensor([0.95])
better_loss = (better_prediction - actual) ** 2
print(better_loss)
# tensor([0.0025]) ← Much lower! Better!
```

**Manual calculation:**

```yaml
Actual: 1.0
Prediction: 0.7
Difference: 0.7 - 1.0 = -0.3
Squared: (-0.3)² = 0.09
Loss: 0.09

Better prediction: 0.95
Difference: 0.95 - 1.0 = -0.05
Squared: (-0.05)² = 0.0025
Loss: 0.0025 ← Much better!
```

## Common Loss Functions

### Mean Squared Error (MSE)

For regression (predicting numbers):

```python
import torch
import torch.nn as nn

# Multiple predictions
predictions = torch.tensor([2.5, 3.1, 4.8])
actual = torch.tensor([2.0, 3.0, 5.0])

# MSE Loss
mse_loss = nn.MSELoss()
loss = mse_loss(predictions, actual)

print(loss)
# tensor(0.1000)

# Manual: ((2.5-2)² + (3.1-3)² + (4.8-5)²) / 3
#       = (0.25 + 0.01 + 0.04) / 3
#       = 0.1
```

### Binary Cross Entropy (BCE)

For binary classification (yes/no):

```python
# Predictions (probabilities)
predictions = torch.tensor([0.9, 0.2, 0.7])

# Actual labels (0 or 1)
labels = torch.tensor([1.0, 0.0, 1.0])

# BCE Loss
bce_loss = nn.BCELoss()
loss = bce_loss(predictions, labels)

print(loss)
# Low loss because predictions are close to labels!
```

### Cross Entropy Loss

For multi-class classification:

```python
# Raw logits (before softmax)
logits = torch.tensor([[2.0, 1.0, 0.1]])

# Actual class (class 0)
target = torch.tensor([0])

# Cross Entropy (includes softmax)
ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(logits, target)

print(loss)
# Lower loss because logits[0]=2.0 is highest!
```

## Why We Minimize Loss

**Goal of training: Make loss as small as possible!**

```yaml
High loss:
  Model is very wrong
  Predictions far from truth
  Need to adjust weights

Low loss:
  Model is accurate
  Predictions close to truth
  Weights are good!

Training:
  Start: High loss (random weights)
  Process: Adjust weights to reduce loss
  End: Low loss (trained model)
```

## Practical Example

```python
import torch
import torch.nn as nn

# Simple model
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

# Data
inputs = torch.tensor([[1.0, 2.0]])
target = torch.tensor([[1.0]])  # Actual answer

# Forward pass
prediction = model(inputs)
print(f"Prediction: {prediction.item():.3f}")

# Calculate loss
loss_fn = nn.BCELoss()
loss = loss_fn(prediction, target)
print(f"Loss: {loss.item():.3f}")

# Interpretation
if loss < 0.1:
    print("Great! Model is accurate")
elif loss < 0.5:
    print("OK, but needs improvement")
else:
    print("Bad! Model needs more training")
```

## Loss Guides Learning

```python
# Loss tells us which direction to adjust weights

# Current prediction vs target
prediction = 0.3
target = 1.0
loss = (prediction - target) ** 2  # 0.49

# If we increase weight:
# prediction becomes 0.6
# loss becomes (0.6 - 1.0)² = 0.16 ← Better!

# If we decrease weight:
# prediction becomes 0.1  
# loss becomes (0.1 - 1.0)² = 0.81 ← Worse!

# So we should INCREASE the weight!
```

## Key Takeaways

✓ **Loss = Error:** Measures how wrong predictions are

✓ **Lower is better:** Training minimizes loss

✓ **Different types:** MSE, BCE, CrossEntropy for different tasks

✓ **Guides learning:** Loss tells us how to adjust weights

✓ **Always positive:** Loss is never negative

**Quick Reference:**

```python
# MSE (regression)
loss = nn.MSELoss()(predictions, targets)

# BCE (binary classification)
loss = nn.BCELoss()(predictions, targets)

# CrossEntropy (multi-class)
loss = nn.CrossEntropyLoss()(logits, targets)

# Training loop
for epoch in range(100):
    prediction = model(x)
    loss = loss_fn(prediction, y)
    # ... backprop and update ...
```

**Remember:** Loss is your compass - it guides the model to better predictions! 🎉
