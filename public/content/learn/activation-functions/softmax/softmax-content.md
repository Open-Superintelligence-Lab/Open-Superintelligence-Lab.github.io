---
hero:
  title: "Softmax"
  subtitle: "Multi-class Classification Activation Function"
  tags:
    - "⚡ Activation Functions"
    - "⏱️ 10 min read"
---

Softmax converts raw model outputs (logits) into **probabilities that sum to 1**. Perfect for multi-class classification!

## The Formula

**Softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)**

For each element:
1. Take exponential (e^x)
2. Divide by sum of all exponentials

This ensures all outputs are positive and sum to exactly 1!

## How It Works

![Softmax Transformation](/content/learn/activation-functions/softmax/softmax-transformation.png)

**Example:**

```python
import torch
import torch.nn as nn

# Raw model outputs (logits)
logits = torch.tensor([2.0, 1.0, 0.1])

# Apply softmax
softmax = nn.Softmax(dim=0)
probabilities = softmax(logits)

print(probabilities)
# tensor([0.6590, 0.2424, 0.0986])

print(probabilities.sum())
# tensor(1.0000) ← Sums to 1!
```

**Manual calculation:**

```yaml
Step 1: Exponentiate each value
  exp(2.0) = 7.389
  exp(1.0) = 2.718
  exp(0.1) = 1.105

Step 2: Sum all exponentials
  Sum = 7.389 + 2.718 + 1.105 = 11.212

Step 3: Divide each by sum
  7.389 / 11.212 = 0.659 (65.9%)
  2.718 / 11.212 = 0.242 (24.2%)
  1.105 / 11.212 = 0.099 (9.9%)

Result: [0.659, 0.242, 0.099]
Verification: 0.659 + 0.242 + 0.099 = 1.0 ✓
```

## Multi-Class Classification

Softmax's main use: **predicting probabilities across multiple classes**!

![Softmax Classification](/content/learn/activation-functions/softmax/softmax-classification.png)

**Example:**

```python
import torch
import torch.nn as nn

# 10-class classification model
class MultiClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)   # Hidden layer
        self.fc3 = nn.Linear(64, 10)    # Output: 10 classes
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        probabilities = self.softmax(logits)  # ← Softmax!
        return probabilities

# Test
model = MultiClassifier()
batch = torch.randn(5, 784)  # 5 images
probs = model(batch)

print(probs.shape)  # torch.Size([5, 10])
print(probs[0])     # First image probabilities
# tensor([0.0823, 0.1245, 0.0567, 0.3421, 0.0912, 
#         0.0734, 0.1823, 0.0234, 0.0156, 0.0085])

print(probs[0].sum())  # tensor(1.0000) ← Sums to 1!

# Get predictions
predictions = torch.argmax(probs, dim=1)
print(predictions)  # tensor([3, 7, 2, 3, 1])
# Class indices with highest probability
```

## Why Exponential?

The exponential makes softmax **sensitive to large values**:

```python
import torch

# Small difference in logits
logits1 = torch.tensor([1.0, 1.1, 1.2])
probs1 = torch.softmax(logits1, dim=0)
print(probs1)
# tensor([0.3006, 0.3322, 0.3672])
# Similar probabilities

# Large difference in logits
logits2 = torch.tensor([1.0, 2.0, 3.0])
probs2 = torch.softmax(logits2, dim=0)
print(probs2)
# tensor([0.0900, 0.2447, 0.6652])
# Clear winner!

# Huge difference
logits3 = torch.tensor([1.0, 5.0, 10.0])
probs3 = torch.softmax(logits3, dim=0)
print(probs3)
# tensor([0.0000, 0.0067, 0.9933])
# Dominant class!
```

**What happened:**

```yaml
exp() amplifies differences:
  
Small logits [1.0, 1.1, 1.2]:
  exp → [2.7, 3.0, 3.3]
  Difference is small → similar probabilities

Large logits [1.0, 5.0, 10.0]:
  exp → [2.7, 148, 22026]
  Difference is HUGE → one dominates
```

## In Code (Simple Implementation)

```python
import torch

def softmax(x):
    """Simple softmax implementation"""
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum()

# Test it
logits = torch.tensor([2.0, 1.0, 0.5])
output = softmax(logits)
print(output)
# tensor([0.6364, 0.2341, 0.1295])
print(output.sum())
# tensor(1.0000) ← Sums to 1!
```

## Using Softmax in PyTorch

### Method 1: As a Layer

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)  # ← Softmax on output
)
```

### Method 2: As a Function

```python
import torch.nn.functional as F

logits = torch.randn(32, 10)  # Batch of 32, 10 classes
probs = F.softmax(logits, dim=1)  # Softmax across classes

print(probs.shape)  # torch.Size([32, 10])
print(probs.sum(dim=1))  # All 1.0
```

### Method 3: Combined with Loss (CrossEntropy)

**Important:** PyTorch's `CrossEntropyLoss` includes softmax!

```python
import torch
import torch.nn as nn

# CrossEntropy already has softmax!
criterion = nn.CrossEntropyLoss()

# Model outputs raw logits (NO softmax)
logits = model(x)
loss = criterion(logits, targets)  # Softmax applied internally!

# DON'T do this:
# probs = F.softmax(logits, dim=1)  # ← Wrong!
# loss = criterion(probs, targets)   # ← Applies softmax twice!
```

## Temperature Scaling

You can control softmax "confidence" with temperature:

```python
import torch

logits = torch.tensor([2.0, 1.0, 0.5])

# Normal softmax (temperature = 1)
probs_normal = torch.softmax(logits, dim=0)
print(probs_normal)
# tensor([0.6364, 0.2341, 0.1295])

# Low temperature (sharper, more confident)
probs_sharp = torch.softmax(logits / 0.5, dim=0)
print(probs_sharp)
# tensor([0.8360, 0.1131, 0.0508])

# High temperature (softer, less confident)
probs_soft = torch.softmax(logits / 2.0, dim=0)
print(probs_soft)
# tensor([0.4750, 0.3107, 0.2143])
```

**Effect of temperature:**

```yaml
T < 1 (low):
  - Sharper probabilities
  - More confident predictions
  - Winner takes more

T > 1 (high):
  - Softer probabilities
  - Less confident predictions
  - More uniform distribution

T = 1:
  - Standard softmax
```

## Practical Example: Image Classification

```python
import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... more layers ...
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
            # NO softmax here if using CrossEntropyLoss!
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        logits = self.classifier(x)
        return logits  # Return logits, not probabilities!

# For inference, apply softmax manually
model = ImageClassifier()
image = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    logits = model(image)
    probs = torch.softmax(logits, dim=1)
    
    # Get top-5 predictions
    top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
    
    print("Top 5 predictions:")
    for i in range(5):
        print(f"Class {top5_indices[0, i]}: {top5_probs[0, i]:.1%}")
```

## Softmax Across Different Dimensions

```python
import torch

# Batch of logits
logits = torch.tensor([[2.0, 1.0, 0.5],
                       [0.8, 2.1, 1.3]])  # 2 samples, 3 classes

# Softmax across classes (dim=1)
probs = torch.softmax(logits, dim=1)
print(probs)
# tensor([[0.6364, 0.2341, 0.1295],
#         [0.1899, 0.6841, 0.1260]])

print(probs.sum(dim=1))  # tensor([1., 1.])
# Each row sums to 1!

# Softmax across samples (dim=0) - unusual!
probs_dim0 = torch.softmax(logits, dim=0)
print(probs_dim0.sum(dim=0))  # tensor([1., 1., 1.])
# Each column sums to 1
```

**Rule:** Use `dim=1` for batch processing (softmax across classes for each sample)!

## Common Mistakes

### ❌ Mistake 1: Softmax Before CrossEntropyLoss

```python
# WRONG - softmax applied twice!
logits = model(x)
probs = torch.softmax(logits, dim=1)
loss = nn.CrossEntropyLoss()(probs, targets)  # ← ERROR!

# CORRECT - CrossEntropy includes softmax
logits = model(x)
loss = nn.CrossEntropyLoss()(logits, targets)  # ← Correct!
```

### ❌ Mistake 2: Wrong Dimension

```python
# Logits shape: (batch_size, num_classes)
logits = torch.randn(32, 10)

# WRONG - softmax across batch
probs = torch.softmax(logits, dim=0)  # ← Each class sums to 1 (weird!)

# CORRECT - softmax across classes
probs = torch.softmax(logits, dim=1)  # ← Each sample sums to 1
```

## Key Takeaways

✓ **Converts to probabilities:** All outputs between 0 and 1

✓ **Sums to 1:** All probabilities add up to exactly 1

✓ **Multi-class:** For 3+ classes (cat, dog, bird, etc.)

✓ **Amplifies differences:** exp() makes large logits dominate

✓ **CrossEntropy includes it:** Don't apply softmax before loss!

✓ **Use dim=1:** For batch processing (softmax per sample)

**Quick Reference:**

```python
# Using softmax
import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Module
softmax_layer = nn.Softmax(dim=1)
probs = softmax_layer(logits)

# Method 2: Functional (most common)
probs = F.softmax(logits, dim=1)

# Method 3: Direct
probs = torch.softmax(logits, dim=1)

# For training with CrossEntropyLoss
criterion = nn.CrossEntropyLoss()  # Includes softmax!
loss = criterion(logits, targets)   # Don't softmax first!

# For inference
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    prediction = torch.argmax(probs, dim=1)
```

**When to use Softmax:**
- ✓ Multi-class classification output (3+ classes)
- ✓ When you need probability distribution
- ✓ Attention mechanisms
- ✗ Binary classification (use sigmoid instead)
- ✗ Regression (use linear output)

**Remember:** Softmax for multi-class, Sigmoid for binary! 🎉
