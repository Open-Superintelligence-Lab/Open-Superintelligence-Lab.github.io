---
hero:
  title: "Backpropagation"
  subtitle: "The Algorithm That Enables Learning"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 18 min read"
---

# Backpropagation

## What is Backpropagation?

Backpropagation (short for "backward propagation of errors") is the algorithm used to **calculate gradients** of the loss function with respect to the weights. It works backward through the network, computing how much each weight contributed to the error.

Think of it as **tracing blame backward** through the network!

![Backpropagation Overview](backprop-overview.png)

## Why It Matters

Without backpropagation:
- ❌ We couldn't efficiently train deep neural networks
- ❌ Would need to compute millions of partial derivatives manually
- ❌ Training would take forever

With backpropagation:
- ✅ Efficiently computes all gradients in one backward pass
- ✅ Uses the chain rule to reuse computations
- ✅ Makes deep learning practical

## The Core Idea

The key insight is the **chain rule** from calculus:

```
If y = f(g(x)), then:
dy/dx = (dy/dg) × (dg/dx)
```

In a neural network with multiple layers, we chain these derivatives together:

```
∂L/∂w⁽¹⁾ = (∂L/∂a⁽³⁾) × (∂a⁽³⁾/∂a⁽²⁾) × (∂a⁽²⁾/∂a⁽¹⁾) × (∂a⁽¹⁾/∂w⁽¹⁾)
```

## The Backpropagation Process

### Step 1: Forward Pass
First, do a forward pass to get the prediction and cache all intermediate values:

```python
# Forward pass (saving values for backprop)
z1 = W1 @ X + b1
a1 = relu(z1)          # Cache z1, a1

z2 = W2 @ a1 + b2
a2 = sigmoid(z2)       # Cache z2, a2 (prediction)

# Compute loss
loss = (a2 - y)**2     # MSE loss
```

### Step 2: Output Layer Gradient
Calculate gradient at the output:

```python
# For MSE loss: L = (ŷ - y)²
dL_da2 = 2 * (a2 - y)

# Gradient through sigmoid
da2_dz2 = a2 * (1 - a2)  # sigmoid derivative

# Combine using chain rule
dL_dz2 = dL_da2 * da2_dz2
```

### Step 3: Propagate Backward
For each layer (from output to input):

```python
# Gradients for layer 2 weights and bias
dL_dW2 = dL_dz2 @ a1.T
dL_db2 = dL_dz2

# Gradient flowing to previous layer
dL_da1 = W2.T @ dL_dz2

# Gradient through ReLU
da1_dz1 = (z1 > 0).astype(float)  # ReLU derivative
dL_dz1 = dL_da1 * da1_dz1

# Gradients for layer 1 weights and bias
dL_dW1 = dL_dz1 @ X.T
dL_db1 = dL_dz1
```

### Step 4: Update Weights
Use gradients to update parameters:

```python
# Gradient descent
learning_rate = 0.01

W1 -= learning_rate * dL_dW1
b1 -= learning_rate * dL_db1
W2 -= learning_rate * dL_dW2
b2 -= learning_rate * dL_db2
```

![Backprop Steps](backprop-steps.png)

## Detailed Example

Let's work through a concrete example with numbers.

### Setup
```
Input: x = 2
Target: y = 1

Network:
- Layer 1: 1 neuron, ReLU
  W1 = 0.5, b1 = 0.1
- Layer 2: 1 neuron, Sigmoid
  W2 = 0.8, b2 = 0.2

Loss: MSE = (ŷ - y)²
```

### Forward Pass
```
Layer 1:
z1 = 0.5(2) + 0.1 = 1.1
a1 = ReLU(1.1) = 1.1

Layer 2:
z2 = 0.8(1.1) + 0.2 = 1.08
a2 = sigmoid(1.08) = 0.746

Loss:
L = (0.746 - 1)² = 0.0645
```

### Backward Pass

**Output Layer:**
```
dL/da2 = 2(0.746 - 1) = -0.508

sigmoid'(z2) = a2(1 - a2)
             = 0.746(1 - 0.746) = 0.189

dL/dz2 = -0.508 × 0.189 = -0.096

dL/dW2 = dL/dz2 × a1 = -0.096 × 1.1 = -0.106
dL/db2 = dL/dz2 = -0.096
```

**Hidden Layer:**
```
dL/da1 = W2 × dL/dz2
       = 0.8 × (-0.096) = -0.077

ReLU'(z1) = 1 (since z1 = 1.1 > 0)

dL/dz1 = -0.077 × 1 = -0.077

dL/dW1 = dL/dz1 × x = -0.077 × 2 = -0.154
dL/db1 = dL/dz1 = -0.077
```

### Update Weights (α = 0.1)
```
W1_new = 0.5 - 0.1(-0.154) = 0.515
b1_new = 0.1 - 0.1(-0.077) = 0.108
W2_new = 0.8 - 0.1(-0.106) = 0.811
b2_new = 0.2 - 0.1(-0.096) = 0.210
```

The weights moved in the direction to reduce the loss! ✅

## Activation Function Derivatives

### ReLU
```python
def relu_derivative(z):
    return (z > 0).astype(float)

# Examples:
relu'(-1) = 0
relu'(0)  = 0
relu'(1)  = 1
```

### Sigmoid
```python
def sigmoid_derivative(a):
    # a is the sigmoid output
    return a * (1 - a)

# Examples:
# If sigmoid(z) = 0.7, then sigmoid'(z) = 0.7 × 0.3 = 0.21
```

### Tanh
```python
def tanh_derivative(a):
    # a is the tanh output
    return 1 - a**2

# Examples:
# If tanh(z) = 0.5, then tanh'(z) = 1 - 0.25 = 0.75
```

### Softmax (special case)
```python
# For softmax with cross-entropy loss, the gradient simplifies to:
dL/dz = a - y  # where a is softmax output, y is one-hot label
```

## Loss Function Gradients

### Mean Squared Error (MSE)
```python
# L = (ŷ - y)²
dL/dŷ = 2(ŷ - y)
```

### Binary Cross-Entropy
```python
# L = -[y log(ŷ) + (1-y)log(1-ŷ)]
dL/dŷ = -(y/ŷ) + (1-y)/(1-ŷ)

# Simplified with sigmoid: dL/dz = ŷ - y
```

### Categorical Cross-Entropy
```python
# L = -Σ yᵢ log(ŷᵢ)
dL/dŷᵢ = -yᵢ/ŷᵢ

# Simplified with softmax: dL/dzᵢ = ŷᵢ - yᵢ
```

## Matrix Form (Batch Processing)

For a batch of examples:

```python
# Forward pass
Z1 = X @ W1.T + b1        # (batch_size, hidden_dim)
A1 = relu(Z1)

Z2 = A1 @ W2.T + b2       # (batch_size, output_dim)
A2 = sigmoid(Z2)

# Loss (averaged over batch)
L = ((A2 - Y)**2).mean()

# Backward pass
dL_dZ2 = (A2 - Y) / batch_size
dL_dW2 = dL_dZ2.T @ A1
dL_db2 = dL_dZ2.sum(axis=0)

dL_dA1 = dL_dZ2 @ W2
dL_dZ1 = dL_dA1 * (Z1 > 0)
dL_dW1 = dL_dZ1.T @ X
dL_db1 = dL_dZ1.sum(axis=0)
```

![Matrix Backprop](matrix-backprop.png)

## Common Challenges

### 1. Vanishing Gradients

**Problem:** Gradients become very small in deep networks

```
# With sigmoid, if all gradients are < 1:
grad = 0.25 × 0.25 × 0.25 × ... → ≈ 0
```

**Solutions:**
- Use ReLU instead of sigmoid/tanh
- Batch normalization
- Residual connections (skip connections)
- Careful weight initialization

### 2. Exploding Gradients

**Problem:** Gradients become very large

```
# If weights are > 1:
grad = 2 × 2 × 2 × ... → ∞
```

**Solutions:**
- Gradient clipping
- Smaller learning rate
- Better weight initialization

### 3. Dead ReLU

**Problem:** ReLU neurons output 0 for all inputs (gradient always 0)

**Solutions:**
- Use Leaky ReLU or ELU
- Lower learning rate
- Better initialization

## Computational Efficiency

Why backpropagation is efficient:

1. **Reuses Computations**
   ```
   ∂L/∂w⁽¹⁾ needs ∂L/∂a⁽²⁾
   ∂L/∂w⁽²⁾ also needs ∂L/∂a⁽²⁾
   → Compute once, use twice!
   ```

2. **One Backward Pass**
   - Forward: O(n) operations
   - Backward: O(n) operations
   - Total: O(2n) ≈ O(n)

3. **Automatic Differentiation**
   - Modern frameworks (PyTorch, TensorFlow) do this automatically
   - Just specify the loss, backprop is automatic!

## PyTorch Example

Here's how easy it is with PyTorch:

```python
import torch
import torch.nn as nn

# Define network
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1),
    nn.Sigmoid()
)

# Forward pass
x = torch.tensor([[2.0, 3.0]])
y = torch.tensor([[1.0]])
y_pred = model(x)

# Compute loss
loss = ((y_pred - y)**2).mean()

# Backward pass (automatic!)
loss.backward()

# Gradients are computed automatically
for name, param in model.named_parameters():
    print(f"{name}: {param.grad}")
```

## Key Takeaways

✅ Backpropagation efficiently computes gradients using the chain rule  
✅ It works backward from output to input layer  
✅ Each layer computes: gradients for weights + gradients for previous layer  
✅ Modern frameworks automate this process  
✅ Understanding it helps with debugging and designing better networks

## What's Next?

Now that we know how to compute gradients, we need to learn how to **use them effectively** to train neural networks. That's where **optimization algorithms** come in!

Let's explore training and optimization next! 🚀

