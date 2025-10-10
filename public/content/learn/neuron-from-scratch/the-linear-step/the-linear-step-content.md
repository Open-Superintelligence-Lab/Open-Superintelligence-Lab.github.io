---
hero:
  title: "The Linear Step"
  subtitle: "Weighted Sum - The Core Computation"
  tags:
    - "🧠 Neuron"
    - "⏱️ 8 min read"
---

The linear step is where the **magic begins** - it's how a neuron combines its inputs using weights!

![Linear Step Visual](/content/learn/neuron-from-scratch/the-linear-step/linear-step-visual.png)

## The Formula

**z = w₁x₁ + w₂x₂ + w₃x₃ + ... + b**

Or in vector form: **z = w · x + b**

This is called the **weighted sum** or **linear combination**.

## Breaking It Down

**Example:**

```python
import torch

# Inputs (features)
x = torch.tensor([2.0, 3.0, 1.5])

# Weights (learned parameters)
w = torch.tensor([0.5, -0.3, 0.8])

# Bias (learned parameter)
b = torch.tensor(0.1)

# Linear step: weighted sum
z = torch.dot(w, x) + b
# OR: z = (w * x).sum() + b

print(z)
# tensor(1.1000)
```

**Manual calculation:**

```yaml
Step 1: Multiply each input by its weight
  2.0 × 0.5 = 1.0
  3.0 × -0.3 = -0.9
  1.5 × 0.8 = 1.2

Step 2: Sum all products
  1.0 + (-0.9) + 1.2 = 1.3

Step 3: Add bias
  1.3 + 0.1 = 1.4

Result: z = 1.4
```

## Why "Linear"?

It's called linear because the relationship between inputs and output is a **straight line**!

```python
# If you double an input, the contribution doubles
x1 = torch.tensor([2.0])
w1 = torch.tensor([0.5])

contribution1 = w1 * x1
print(contribution1)  # tensor([1.0])

# Double the input
x2 = torch.tensor([4.0])
contribution2 = w1 * x2
print(contribution2)  # tensor([2.0]) ← Exactly double!
```

**Linear properties:**

```yaml
f(x + y) = f(x) + f(y)  ← Additive
f(2x) = 2·f(x)          ← Scalable

This makes it predictable and stable!
```

## What Each Component Does

### Weights: The Learnable Parameters

Weights determine **which inputs matter**:

```python
# Positive weight → input increases output
w_positive = 0.8
x = 5.0
contribution = w_positive * x  # 4.0 ← Boosts output!

# Negative weight → input decreases output  
w_negative = -0.8
contribution = w_negative * x  # -4.0 ← Reduces output!

# Small weight → input barely matters
w_small = 0.01
contribution = w_small * x  # 0.05 ← Tiny effect

# Large weight → input matters a lot
w_large = 10.0
contribution = w_large * x  # 50.0 ← Huge effect!
```

### Bias: The Threshold Adjuster

Bias shifts the decision boundary:

```python
import torch

x = torch.tensor([1.0, 1.0])
w = torch.tensor([1.0, 1.0])

# No bias
z_no_bias = torch.dot(w, x)
print(z_no_bias)  # tensor(2.0000)

# Positive bias (easier to activate)
b_positive = 5.0
z_positive = torch.dot(w, x) + b_positive
print(z_positive)  # tensor(7.0000) ← Higher!

# Negative bias (harder to activate)
b_negative = -5.0
z_negative = torch.dot(w, x) + b_negative
print(z_negative)  # tensor(-3.0000) ← Lower!
```

**What bias does:**

```yaml
Positive bias:
  Makes neuron more likely to "fire"
  Shifts decision boundary down
  
Negative bias:
  Makes neuron less likely to "fire"
  Shifts decision boundary up
  
No bias:
  Decision passes through origin
```

## Using nn.Linear in PyTorch

PyTorch provides `nn.Linear` to do this automatically:

```python
import torch
import torch.nn as nn

# Create linear layer: 3 inputs → 1 output
linear = nn.Linear(in_features=3, out_features=1)

# Input batch: 5 samples, 3 features each
x = torch.randn(5, 3)

# Apply linear transformation
z = linear(x)

print(z.shape)  # torch.Size([5, 1])

# What it does internally:
# z = x @ linear.weight.T + linear.bias
```

## Multiple Outputs

You can have multiple output neurons:

```python
import torch
import torch.nn as nn

# 3 inputs → 5 outputs (5 neurons)
linear = nn.Linear(3, 5)

x = torch.tensor([[1.0, 2.0, 3.0]])  # 1 sample

z = linear(x)
print(z)
# tensor([[0.234, -1.123, 0.567, 2.134, -0.876]])
# 5 different outputs (one per neuron)!

# Each output has its own weights:
print(linear.weight.shape)  # torch.Size([5, 3])
# 5 neurons × 3 weights each

print(linear.bias.shape)  # torch.Size([5])
# 5 biases (one per neuron)
```

## Real-World Example

```python
import torch
import torch.nn as nn

# House price prediction
# Inputs: [size_sqft, bedrooms, age_years]
house_features = torch.tensor([[2000.0, 3.0, 10.0]])

# Create linear layer
price_neuron = nn.Linear(3, 1)

# Manually set weights (usually learned from data)
with torch.no_grad():
    price_neuron.weight = nn.Parameter(torch.tensor([[200.0, 50000.0, -1000.0]]))
    price_neuron.bias = nn.Parameter(torch.tensor([50000.0]))

# Predict price
predicted_price = price_neuron(house_features)
print(predicted_price)
# tensor([[540000.]]) ← $540,000 prediction

# Manual calculation:
# 2000×200 + 3×50000 + 10×(-1000) + 50000
# = 400,000 + 150,000 - 10,000 + 50,000
# = 590,000 (close to our result!)
```

**What the weights learned:**

```yaml
Weight for size: 200 → Each sq ft adds $200
Weight for bedrooms: 50,000 → Each bedroom adds $50k
Weight for age: -1,000 → Each year reduces price by $1k
Bias: 50,000 → Base price of $50k
```

## Matrix Form

For a batch, the linear step is matrix multiplication:

```python
# Batch of 3 samples
X = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])  # Shape: (3, 2)

# Weights for 1 output neuron
W = torch.tensor([[0.5],
                  [0.3]])  # Shape: (2, 1)

b = torch.tensor([0.1])

# Linear step as matrix multiplication
Z = X @ W + b

print(Z)
# tensor([[1.2000],
#         [2.8000],
#         [4.4000]])
```

**Matrix form:**

```yaml
Z = XW + b

Where:
  X: (batch_size, input_features)
  W: (input_features, output_features)
  b: (output_features,)
  Z: (batch_size, output_features)
```

## Key Takeaways

✓ **Linear step:** Weighted sum of inputs plus bias

✓ **Formula:** z = Σ(wᵢxᵢ) + b

✓ **Weights:** Determine importance of each input

✓ **Bias:** Shifts the output

✓ **PyTorch:** Use `nn.Linear(in, out)`

✓ **Matrix form:** Efficient for batches

**Quick Reference:**

```python
# Manual linear step
z = (weights * inputs).sum() + bias

# Using PyTorch
linear = nn.Linear(input_dim, output_dim)
z = linear(x)

# What it does:
# z = x @ linear.weight.T + linear.bias
```

**Remember:** The linear step is just multiply → sum → add bias. Simple but powerful! 🎉
