---
hero:
  title: "ReLU"
  subtitle: "Rectified Linear Unit - The Most Popular Activation Function"
  tags:
    - "⚡ Activation Functions"
    - "⏱️ 10 min read"
---

ReLU is the **most widely used** activation function in deep learning. It's simple, fast, and works incredibly well!

## The Formula

**ReLU(x) = max(0, x)**

That's it! If the input is negative, output 0. If positive, output the input unchanged.

![ReLU Graph](/content/learn/activation-functions/relu/relu-graph.png)

```yaml
Input < 0  →  Output = 0
Input ≥ 0  →  Output = Input

Examples:
ReLU(-5) = 0
ReLU(-1) = 0
ReLU(0) = 0
ReLU(3) = 3
ReLU(10) = 10
```

## How It Works

**Example:**

```python
import torch
import torch.nn as nn

# Create ReLU activation
relu = nn.ReLU()

# Test with different values
x = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0])
output = relu(x)

print(output)
# tensor([0., 0., 0., 2., 5.])
```

**Manual calculation:**

```yaml
Input:   [-3.0, -1.0,  0.0,  2.0,  5.0]
          ↓     ↓      ↓     ↓     ↓
ReLU:    max(0,-3) max(0,-1) max(0,0) max(0,2) max(0,5)
          ↓     ↓      ↓     ↓     ↓
Output:  [0.0,  0.0,   0.0,  2.0,  5.0]
```

![ReLU Example](/content/learn/activation-functions/relu/relu-example.png)

**The rule:** Negative numbers get "zeroed out", positive numbers pass through unchanged.

## In Code (Simple Implementation)

You can implement ReLU yourself:

```python
import torch

def relu(x):
    """Simple ReLU implementation"""
    return torch.maximum(torch.tensor(0.0), x)

# Test it
x = torch.tensor([-2.0, 3.0, -1.0, 4.0])
output = relu(x)
print(output)
# tensor([0., 3., 0., 4.])
```

Or even simpler with element-wise operations:

```python
def relu_simple(x):
    """Even simpler ReLU"""
    return x * (x > 0)  # Multiply by boolean mask

x = torch.tensor([-2.0, 3.0, -1.0, 4.0])
output = relu_simple(x)
print(output)
# tensor([0., 3., 0., 4.])
```

## Why ReLU is Amazing

### 1. Simple and Fast

```yaml
Computation: Just one comparison!
  if x > 0: return x
  else: return 0

No expensive operations:
  ✓ No exponentials (unlike sigmoid/tanh)
  ✓ No divisions
  ✓ Just comparison and selection
```

### 2. Solves Vanishing Gradient Problem

For positive values, gradient is always 1:

```python
import torch

x = torch.tensor([5.0], requires_grad=True)
y = torch.relu(x)
y.backward()

print(x.grad)  # tensor([1.])
# Gradient is 1 for positive inputs!
```

**Why this matters:**

```yaml
Sigmoid/Tanh: gradients get very small (vanishing)
ReLU: gradient is 1 for positive inputs

Result: Faster training, deeper networks possible!
```

### 3. Creates Sparsity

ReLU zeros out negative values, creating sparse activations:

![ReLU Network](/content/learn/activation-functions/relu/relu-network.png)

```python
# Example: network layer output
layer_output = torch.tensor([-2.1, 3.5, -0.8, 1.2, -1.5])
activated = torch.relu(layer_output)

print(activated)
# tensor([0.0, 3.5, 0.0, 1.2, 0.0])

# 60% of activations are zero!
sparsity = (activated == 0).sum().item() / activated.numel()
print(f"Sparsity: {sparsity:.1%}")
# Output: Sparsity: 60.0%
```

**Benefits of sparsity:**

```yaml
Sparse networks:
  ✓ More efficient (many zeros)
  ✓ Better generalization
  ✓ Easier to interpret
  ✓ Faster computation
```

## Using ReLU in PyTorch

### Method 1: As a Layer

```python
import torch.nn as nn

# Create a neural network with ReLU
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),           # ← ReLU activation
    nn.Linear(20, 5),
    nn.ReLU(),           # ← Another ReLU
    nn.Linear(5, 1)
)
```

### Method 2: As a Function

```python
import torch
import torch.nn.functional as F

x = torch.randn(5, 10)

# Apply ReLU directly
output = F.relu(x)

# Same as
output = torch.relu(x)
```

### Method 3: Manual Implementation

```python
# In your custom forward pass
def forward(self, x):
    x = self.linear1(x)
    x = torch.relu(x)      # Apply ReLU
    x = self.linear2(x)
    return x
```

## Practical Example: Multi-Layer Network

```python
import torch
import torch.nn as nn

# 3-layer network with ReLU
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)  # Input layer
        self.fc2 = nn.Linear(256, 128)  # Hidden layer
        self.fc3 = nn.Linear(128, 10)   # Output layer
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)  # ReLU after layer 1
        
        x = self.fc2(x)
        x = torch.relu(x)  # ReLU after layer 2
        
        x = self.fc3(x)
        # No ReLU on output layer!
        return x

# Test it
model = SimpleNet()
input_data = torch.randn(32, 784)  # Batch of 32
output = model(input_data)

print(output.shape)  # torch.Size([32, 10])
```

## The Dying ReLU Problem

**Issue:** Sometimes neurons can get "stuck" outputting only zeros.

```python
# Neuron with large negative bias
weights = torch.randn(10)
bias = torch.tensor(-100.0)  # Very negative!

# Forward pass
x = torch.randn(10)
linear_output = x @ weights + bias
activated = torch.relu(linear_output)

print(linear_output)  # tensor(-98.5) - always negative!
print(activated)      # tensor(0.) - always zero!
```

**Why this happens:**

```yaml
1. Neuron produces negative output
2. ReLU makes it zero
3. Gradient for negative inputs is also zero
4. Neuron never updates → stuck at zero forever!

Solution: Use variants like Leaky ReLU or careful initialization
```

## ReLU Variants

### Leaky ReLU

Allows small negative values:

```python
import torch.nn as nn

# Standard ReLU
relu = nn.ReLU()
print(relu(torch.tensor(-1.0)))  # tensor(0.)

# Leaky ReLU (small slope for negatives)
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
print(leaky_relu(torch.tensor(-1.0)))  # tensor(-0.0100)
```

**Formula:**

```yaml
LeakyReLU(x) = max(0.01x, x)

For x < 0: output = 0.01 * x (small negative)
For x ≥ 0: output = x (unchanged)
```

## Key Takeaways

✓ **Simple formula:** max(0, x)

✓ **Fast:** Just comparison, no complex math

✓ **Solves vanishing gradients:** Gradient is 1 for positive values

✓ **Creates sparsity:** Zeros out negative activations

✓ **Most popular:** Default choice for hidden layers

✓ **Watch out for:** Dying ReLU (neurons stuck at zero)

**Quick Reference:**

```python
# Using ReLU
import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Module
relu_layer = nn.ReLU()
output = relu_layer(x)

# Method 2: Functional
output = F.relu(x)

# Method 3: Direct
output = torch.relu(x)

# Method 4: Manual
output = torch.maximum(torch.tensor(0.0), x)
```

**When to use ReLU:**
- ✓ Hidden layers in CNNs
- ✓ Hidden layers in feedforward networks
- ✓ Default activation for most architectures
- ✗ NOT for output layer (use softmax/sigmoid/linear instead)

**Remember:** ReLU is simple but powerful. It's the workhorse of modern deep learning! 🎉
