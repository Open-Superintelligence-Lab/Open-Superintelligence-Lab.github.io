---
hero:
  title: "Sigmoid"
  subtitle: "The Classic S-shaped Activation Function"
  tags:
    - "⚡ Activation Functions"
    - "⏱️ 10 min read"
---

Sigmoid is a smooth, S-shaped function that **squashes any input to a value between 0 and 1**. Perfect for probabilities!

## The Formula

**σ(x) = 1 / (1 + e⁻ˣ)**

The output is always between 0 and 1, making it ideal for binary classification!

![Sigmoid Graph](/content/learn/activation-functions/sigmoid/sigmoid-graph.png)

```yaml
Input → -∞  →  Output → 0
Input = 0   →  Output = 0.5
Input → +∞  →  Output → 1

Key property: Output is always in (0, 1)
```

## How It Works

**Example:**

```python
import torch
import torch.nn as nn

# Create sigmoid activation
sigmoid = nn.Sigmoid()

# Test with different values
x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
output = sigmoid(x)

print(output)
# tensor([0.0067, 0.2689, 0.5000, 0.7311, 0.9933])
```

![Sigmoid Example](/content/learn/activation-functions/sigmoid/sigmoid-example.png)

**Manual calculation (for x = 2):**

```yaml
σ(2) = 1 / (1 + e⁻²)
     = 1 / (1 + 0.1353)
     = 1 / 1.1353
     = 0.881

Result: ~0.88 or 88% probability
```

## The S-Shape Explained

```yaml
Large negative input (x = -10):
  e⁻⁽⁻¹⁰⁾ = e¹⁰ = 22026 (huge!)
  σ(x) = 1 / (1 + 22026) ≈ 0.00005
  → Output near 0

Zero input (x = 0):
  e⁻⁰ = 1
  σ(x) = 1 / (1 + 1) = 0.5
  → Output exactly 0.5

Large positive input (x = 10):
  e⁻¹⁰ = 0.000045 (tiny!)
  σ(x) = 1 / (1 + 0.000045) ≈ 0.99995
  → Output near 1
```

## Binary Classification

Sigmoid's killer application: **predicting probabilities for binary classification**!

![Sigmoid Classification](/content/learn/activation-functions/sigmoid/sigmoid-classification.png)

**Example:**

```python
import torch
import torch.nn as nn

# Binary classification model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)  # 10 features → 1 output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        logits = self.linear(x)
        probabilities = self.sigmoid(logits)
        return probabilities

# Test
model = BinaryClassifier()
x = torch.randn(5, 10)  # 5 samples, 10 features each
probs = model(x)

print(probs)
# tensor([[0.7234],
#         [0.3421],
#         [0.8956],
#         [0.1234],
#         [0.6543]], grad_fn=<SigmoidBackward0>)

# Convert to predictions
predictions = (probs > 0.5).float()
print(predictions)
# tensor([[1.],  # Class 1 (prob > 0.5)
#         [0.],  # Class 0 (prob < 0.5)
#         [1.],
#         [0.],
#         [1.]])
```

**What happened:**

```yaml
Model output (logit): 2.5
     ↓
Sigmoid: 1/(1 + e⁻²·⁵) = 0.92
     ↓
0.92 > 0.5 → Predict Class 1!
```

## In Code (Simple Implementation)

```python
import torch

def sigmoid(x):
    """Simple sigmoid implementation"""
    return 1 / (1 + torch.exp(-x))

# Test it
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = sigmoid(x)
print(output)
# tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
```

## Using Sigmoid in PyTorch

### Method 1: As a Layer

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid()  # ← Output layer for binary classification
)
```

### Method 2: As a Function

```python
import torch
import torch.nn.functional as F

x = torch.randn(5, 1)
output = F.sigmoid(x)  # or torch.sigmoid(x)
```

### Method 3: Combined with Loss (BCE)

```python
import torch
import torch.nn as nn

# Binary Cross Entropy already includes sigmoid!
criterion = nn.BCEWithLogitsLoss()  # Sigmoid + BCE

# Model outputs raw logits (no sigmoid)
logits = model(x)
loss = criterion(logits, targets)  # Sigmoid applied internally!
```

## The Vanishing Gradient Problem

Sigmoid's main weakness: **gradients vanish for large inputs**!

```python
import torch

# Large input
x = torch.tensor([10.0], requires_grad=True)
y = torch.sigmoid(x)
y.backward()

print(f"Input: {x.item()}")
print(f"Output: {y.item():.6f}")
print(f"Gradient: {x.grad.item():.6f}")
# Gradient: 0.000045 ← Very small!
```

**Why this is bad:**

```yaml
Gradient too small →
  Slow learning →
    Deep networks struggle →
      ReLU became more popular!

This is why ReLU replaced sigmoid in hidden layers.
```

**When sigmoid gradients vanish:**

```yaml
For x = -10 or x = 10:
  Output is ~0 or ~1 (saturated)
  Gradient ≈ 0 (flat region)
  Learning stops!

For x near 0:
  Output around 0.5 (steep region)
  Gradient maximum (~0.25)
  Learning is good here
```

## Practical Examples

### Example 1: Email Spam Detector

```python
import torch
import torch.nn as nn

class SpamDetector(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        probability = self.sigmoid(x)  # Sigmoid at end!
        return probability

# Predict
email_features = torch.randn(1, 100)
spam_probability = model(email_features)

if spam_probability > 0.5:
    print(f"SPAM (confidence: {spam_probability.item():.2%})")
else:
    print(f"NOT SPAM (confidence: {1-spam_probability.item():.2%})")
```

### Example 2: Medical Diagnosis

```python
# Patient features → Disease probability
patient = torch.randn(1, 50)  # 50 medical features
probability = model(patient)

print(f"Disease probability: {probability.item():.1%}")
# Output: Disease probability: 23.4%

if probability > 0.7:
    print("High risk - recommend testing")
elif probability > 0.3:
    print("Medium risk - monitor")
else:
    print("Low risk")
```

## Sigmoid vs ReLU

```yaml
Sigmoid:
  ✓ Outputs 0 to 1 (probabilities)
  ✓ Smooth, differentiable everywhere
  ✓ Great for binary classification OUTPUT
  ✗ Vanishing gradients for large |x|
  ✗ Slow computation (exponential)
  ✗ NOT zero-centered

ReLU:
  ✓ Fast (simple comparison)
  ✓ No vanishing gradient for x > 0
  ✓ Creates sparsity
  ✗ Outputs 0 to ∞ (not probabilities)
  ✗ Dying ReLU problem
  ✗ NOT smooth at x = 0
```

**When to use each:**

```yaml
Use Sigmoid for:
  ✓ Binary classification output layer
  ✓ When you need probabilities
  ✓ Gates in LSTM/GRU

Use ReLU for:
  ✓ Hidden layers
  ✓ Convolutional layers
  ✓ Most modern architectures
```

## Key Takeaways

✓ **S-shaped curve:** Smooth transition from 0 to 1

✓ **Formula:** σ(x) = 1 / (1 + e⁻ˣ)

✓ **Output range:** Always between 0 and 1

✓ **Perfect for probabilities:** Binary classification output

✓ **Vanishing gradients:** Problem in deep networks

✓ **Mostly for output:** ReLU used in hidden layers instead

**Quick Reference:**

```python
# Using sigmoid
import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Module
sigmoid_layer = nn.Sigmoid()
output = sigmoid_layer(x)

# Method 2: Functional
output = F.sigmoid(x)

# Method 3: Direct
output = torch.sigmoid(x)

# Method 4: Manual
output = 1 / (1 + torch.exp(-x))

# For binary classification with loss
criterion = nn.BCEWithLogitsLoss()  # Includes sigmoid!
```

**Remember:** Sigmoid for the output, ReLU for the hidden layers! 🎉
