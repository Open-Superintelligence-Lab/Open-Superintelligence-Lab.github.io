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

## The Mathematical Foundation

The linear step performs what mathematicians call an **affine transformation**. It's the fundamental operation in every neural network, repeated millions of times during training.

### Why "Linear"?

In mathematics, a linear transformation has two key properties:

1. **Additivity:** f(x + y) = f(x) + f(y)
2. **Homogeneity:** f(αx) = αf(x)

Our weighted sum (without bias) satisfies both properties. The bias term makes it technically "affine" rather than purely linear, but we still call it the "linear step" in deep learning.

**Geometric interpretation:** The linear step projects the input vector onto a weight vector, then shifts the result by the bias. This creates a **hyperplane** in the input space - a decision boundary!

## The Formula

**z = w₁x₁ + w₂x₂ + w₃x₃ + ... + b**

Or in vector form: **z = w · x + b**

This is called the **weighted sum** or **linear combination**.

### Notation Explained

- **z**: The output (called "pre-activation" or "logit")
- **w₁, w₂, w₃, ...**: Weight parameters (subscript indicates which input)
- **x₁, x₂, x₃, ...**: Input features (subscript indicates position)
- **b**: Bias term (shifts the entire output)
- **·** (dot): The dot product operation Σᵢwᵢxᵢ

### Multiple Mathematical Representations

The same operation can be written in several equivalent ways:

```
Explicit sum:     z = w₁x₁ + w₂x₂ + w₃x₃ + b
Summation notation: z = Σᵢ(wᵢxᵢ) + b
Dot product:      z = w · x + b
Matrix form:      z = w^T x + b
Einstein notation: z = wᵢxᵢ + b  (repeated index implies sum)
```

All represent the exact same computation!

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
# tensor(1.4000)
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

### Vector Operations Breakdown

Let's understand the vector math step-by-step:

**Given:**
```
x = [2.0, 3.0, 1.5]  ← Input vector (3D)
w = [0.5, -0.3, 0.8]  ← Weight vector (3D)
b = 0.1              ← Bias scalar
```

**Dot product calculation:**
```
w · x = Σᵢwᵢxᵢ 
      = w₁x₁ + w₂x₂ + w₃x₃
      = (0.5)(2.0) + (-0.3)(3.0) + (0.8)(1.5)
      = 1.0 + (-0.9) + 1.2
      = 1.3
```

**Add bias:**
```
z = w · x + b
  = 1.3 + 0.1
  = 1.4
```

**Geometric meaning:** The dot product w · x measures how "aligned" the input is with the weight vector. If they point in the same direction, the dot product is large and positive. If they point in opposite directions, it's large and negative.

## Why "Linear"?

It's called linear because the relationship between inputs and output follows the properties of **linearity** in mathematics.

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
f(αx) = α·f(x)          ← Homogeneous (scalable)

This makes it predictable and stable!
```

### Mathematical Proof of Linearity

Let's verify these properties for our weighted sum (ignoring bias for now):

**Property 1: Additivity**
```
f(x + y) = w · (x + y)
         = w · x + w · y    (distributive property of dot product)
         = f(x) + f(y)  ✓
```

**Property 2: Homogeneity**
```
f(αx) = w · (αx)
      = α(w · x)       (scalar multiplication)
      = αf(x)  ✓
```

**Note:** The bias term b breaks pure linearity, making it an "affine" transformation. But we still get the key benefit: **no exponentials, no multiplications between variables** - just weighted sums!

### Why This Matters

Linear operations have crucial properties for machine learning:

1. **Computational efficiency**: Just multiply and add - very fast!
2. **Gradient flow**: Derivatives are constants, making backpropagation stable
3. **Interpretability**: Each weight directly shows importance of its input
4. **Composability**: Can stack linear layers (though we need non-linearity between them)

## What Each Component Does

### Weights: The Learnable Parameters

Weights are the **heart** of machine learning - they're what the model learns! Let's understand them deeply.

**Mathematical role:** Weights define a hyperplane in the input space. The equation w₁x₁ + w₂x₂ + ... + wₙxₙ + b = 0 describes the decision boundary.

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

**Mathematical interpretation of weight magnitudes:**

```
|w| (magnitude):
  |w| < 0.1  → Feature is mostly ignored
  0.1 < |w| < 1.0  → Feature has moderate importance
  |w| > 1.0  → Feature is highly important
  
sign(w) (direction):
  w > 0  → Positive correlation (input ↑ → output ↑)
  w < 0  → Negative correlation (input ↑ → output ↓)
  w = 0  → No relationship
```

### Bias: The Threshold Adjuster

Bias shifts the decision boundary without affecting the direction.

**Mathematical perspective:** In the equation w · x + b = 0, the bias b shifts the hyperplane away from the origin. Without bias, the decision boundary must pass through the origin (0, 0, ..., 0).

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

### The Role of Bias in Different Contexts

**Example 1: Temperature prediction**
```
If inputs are already centered (mean=0), bias represents the average temperature
If inputs are raw values, bias adjusts for the baseline temperature
```

**Example 2: Binary classification**
```
If classes are balanced (50-50), bias should be near 0
If classes are imbalanced (90-10), bias should favor the majority class initially
```

**Mathematical form:**
```
z = w · x + b

Without bias (b=0):
  z = w · x
  Decision boundary: w · x = 0 (passes through origin)
  
With bias:
  z = w · x + b
  Decision boundary: w · x = -b (shifted from origin)
  Distance from origin: |b| / ||w||
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
# tensor([[590000.]]) ← $590,000 prediction

# Manual calculation:
# 2000×200 + 3×50000 + 10×(-1000) + 50000
# = 400,000 + 150,000 - 10,000 + 50,000
# = 590,000 (perfect match!)
```

**What the weights learned:**

```yaml
Weight for size: 200 → Each sq ft adds $200
Weight for bedrooms: 50,000 → Each bedroom adds $50k
Weight for age: -1,000 → Each year reduces price by $1k
Bias: 50,000 → Base price of $50k
```

## Matrix Form

For a batch, the linear step is matrix multiplication. This is where linear algebra becomes essential!

### Understanding the Dimensions

When processing multiple samples at once (a batch), we use matrix operations:

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

### Detailed Matrix Multiplication Breakdown

Let's understand how matrix multiplication works element by element:

```
Given:
X = [[x₁₁, x₁₂],    (2 samples, 2 features)
     [x₂₁, x₂₂]]

W = [[w₁₁],          (2 features, 1 output)
     [w₂₁]]

Result Z = XW:
Z[0,0] = x₁₁·w₁₁ + x₁₂·w₂₁  ← Dot product of row 1 with column 1
Z[1,0] = x₂₁·w₁₁ + x₂₂·w₂₁  ← Dot product of row 2 with column 1
```

**With our numbers:**
```
X = [[1.0, 2.0],
     [3.0, 4.0],
     [5.0, 6.0]]

W = [[0.5],
     [0.3]]

Z = [[1.0×0.5 + 2.0×0.3],   = [[1.1],
     [3.0×0.5 + 4.0×0.3],   =  [2.7],
     [5.0×0.5 + 6.0×0.3]]   =  [4.3]]

Then add bias:
Z + b = [[1.1 + 0.1],   = [[1.2],
         [2.7 + 0.1],   =  [2.8],
         [4.3 + 0.1]]   =  [4.4]]
```

### Broadcasting the Bias

The bias b is broadcast (copied) across all samples in the batch:

```
Z = XW + b

Where b is added to each row:
[[z₁],     [[b],       [[z₁+b],
 [z₂],  +   [b],    =   [z₂+b],
 [z₃]]      [b]]        [z₃+b]]
```

This is automatic in PyTorch and NumPy - the bias is broadcast to match the shape!

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
