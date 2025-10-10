---
hero:
  title: "Forward Propagation"
  subtitle: "How Data Flows Through Neural Networks"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 13 min read"
---

# Forward Propagation

## What is Forward Propagation?

Forward propagation is the process of passing input data through the neural network to get an output (prediction). It's called **"forward"** because data moves in one direction:

```
Input Layer → Hidden Layers → Output Layer
```

This is how neural networks make predictions!

![Forward Propagation Flow](forward-prop-diagram.png)

## The Process Step by Step

### Step 1: Input Layer
Receive the input features

```python
# Example: Image of handwritten digit
x = [0.5, 0.8, 0.3, ...]  # Pixel values
```

### Step 2: Weighted Sum
For each neuron in the next layer, calculate:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Or in matrix form:
```
Z = WX + b
```

Where:
- `W` = weight matrix
- `X` = input vector
- `b` = bias vector

### Step 3: Activation Function
Apply non-linear activation:

```
a = σ(z)  # e.g., ReLU(z) or sigmoid(z)
```

### Step 4: Repeat
Use the outputs as inputs for the next layer, repeat steps 2-3 until reaching the output layer.

## Mathematical Representation

For a network with L layers:

```
Layer 1: a⁽¹⁾ = σ(W⁽¹⁾x + b⁽¹⁾)
Layer 2: a⁽²⁾ = σ(W⁽²⁾a⁽¹⁾ + b⁽²⁾)
...
Layer L: a⁽ᴸ⁾ = σ(W⁽ᴸ⁾a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾)
```

The final output `a⁽ᴸ⁾` is our prediction!

## Simple Example: 2-Layer Network

Let's walk through a tiny network:

**Network Architecture:**
- Input: 2 features
- Hidden layer: 3 neurons (ReLU)
- Output: 1 neuron (Sigmoid)

### Given:
```
Input: x = [2, 3]

Hidden layer weights:
W⁽¹⁾ = [[0.5, 0.3],
        [0.2, 0.8],
        [0.1, 0.6]]

Hidden layer bias: b⁽¹⁾ = [0.1, 0.2, 0.3]

Output layer weights: W⁽²⁾ = [[0.4, 0.5, 0.6]]
Output layer bias: b⁽²⁾ = [0.1]
```

### Step-by-Step Calculation:

**Hidden Layer (Layer 1):**

Neuron 1:
```
z₁⁽¹⁾ = 0.5(2) + 0.3(3) + 0.1 = 2.0
a₁⁽¹⁾ = ReLU(2.0) = 2.0
```

Neuron 2:
```
z₂⁽¹⁾ = 0.2(2) + 0.8(3) + 0.2 = 3.0
a₂⁽¹⁾ = ReLU(3.0) = 3.0
```

Neuron 3:
```
z₃⁽¹⁾ = 0.1(2) + 0.6(3) + 0.3 = 2.3
a₃⁽¹⁾ = ReLU(2.3) = 2.3
```

Hidden layer output: `a⁽¹⁾ = [2.0, 3.0, 2.3]`

**Output Layer (Layer 2):**
```
z⁽²⁾ = 0.4(2.0) + 0.5(3.0) + 0.6(2.3) + 0.1 = 3.68
a⁽²⁾ = sigmoid(3.68) ≈ 0.975
```

**Final Prediction: 0.975** (97.5% probability for class 1)

![Example Network](forward-example.png)

## Matrix Operations (Vectorized)

For efficiency, we compute for all neurons at once:

### Layer 1:
```python
import numpy as np

# Input
X = np.array([2, 3])

# Layer 1
W1 = np.array([[0.5, 0.3],
               [0.2, 0.8],
               [0.1, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])

Z1 = W1 @ X + b1  # Matrix multiplication
A1 = np.maximum(0, Z1)  # ReLU

# Layer 2
W2 = np.array([[0.4, 0.5, 0.6]])
b2 = np.array([0.1])

Z2 = W2 @ A1 + b2
A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid

print(f"Prediction: {A2[0]:.3f}")
# Output: Prediction: 0.975
```

## Batch Processing

In practice, we process **multiple examples** simultaneously:

```python
# Batch of 3 examples
X = np.array([[2, 3],
              [1, 4],
              [3, 2]])  # Shape: (3, 2)

# Forward pass
Z1 = X @ W1.T + b1  # Broadcasting handles bias
A1 = np.maximum(0, Z1)

Z2 = A1 @ W2.T + b2
A2 = 1 / (1 + np.exp(-Z2))

print(A2.shape)  # (3, 1) - predictions for 3 examples
```

## Activation Functions in Action

Different activation functions transform data differently:

### ReLU
```python
def relu(z):
    return np.maximum(0, z)

# Keeps positive values, zeros out negative
relu([-2, -1, 0, 1, 2])  # [0, 0, 0, 1, 2]
```

### Sigmoid
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Squashes to (0, 1)
sigmoid([-2, 0, 2])  # [0.119, 0.5, 0.881]
```

### Tanh
```python
def tanh(z):
    return np.tanh(z)

# Squashes to (-1, 1)
tanh([-2, 0, 2])  # [-0.964, 0, 0.964]
```

![Activation Functions](activations-comparison.png)

## Common Patterns

### Classification (Softmax Output)
For multi-class classification, use softmax in the output layer:

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Numerical stability
    return exp_z / exp_z.sum()

# Example: 3-class classification
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
# [0.659, 0.242, 0.099] - probabilities sum to 1
```

### Regression (Linear Output)
For regression, no activation in output layer:

```python
# Final layer for regression
output = W_last @ a_last + b_last
# No activation - can output any real number
```

## Key Properties

### Deterministic
Same input + same weights = same output every time

### Differentiable
We can compute gradients (needed for backpropagation)

### Composable
Output of one layer is input to next - function composition

### Efficient
Matrix operations are highly optimized (GPUs!)

## Debugging Forward Pass

Common issues and solutions:

### 1. Shape Mismatches
```python
# Check shapes at each layer
print(f"Input shape: {X.shape}")
print(f"W1 shape: {W1.shape}")
print(f"Z1 shape: {Z1.shape}")
```

### 2. Numerical Overflow
```python
# For sigmoid/softmax, use numerical stability tricks
# Bad:  exp(x) / sum(exp(x))
# Good: exp(x - max(x)) / sum(exp(x - max(x)))
```

### 3. Wrong Activation
```python
# Make sure you use the right activation for each layer
# Hidden: ReLU, Tanh
# Output (classification): Sigmoid (binary), Softmax (multi-class)
# Output (regression): None (linear)
```

## Implementation Tips

✅ Use vectorized operations (NumPy/PyTorch)  
✅ Process data in batches for efficiency  
✅ Cache intermediate values (needed for backprop)  
✅ Add assertions to check shapes  
✅ Normalize inputs for stable training

## What We've Learned

🎯 Forward propagation transforms inputs into predictions  
🎯 It's a series of weighted sums + activations  
🎯 Matrix operations make it efficient  
🎯 Different activations serve different purposes  
🎯 The process is deterministic and differentiable  

## Next Steps

Forward propagation gets us predictions, but how does the network **learn**? That's where **backpropagation** comes in! It calculates how to adjust the weights to improve predictions.

Let's dive into backpropagation next! 🎓

