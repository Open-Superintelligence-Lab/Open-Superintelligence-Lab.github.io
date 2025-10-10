---
hero:
  title: "The Activation Function"
  subtitle: "Adding Non-Linearity to Neurons"
  tags:
    - "🧠 Neuron"
    - "⏱️ 8 min read"
---

The activation function is what makes neural networks **powerful**. Without it, you'd just have fancy linear regression!

## Why We Need Activation Functions

**Without activation:** No matter how many layers, it's still just linear!

```python
import torch
import torch.nn as nn

# Network WITHOUT activation functions
model_linear = nn.Sequential(
    nn.Linear(10, 20),
    # No activation!
    nn.Linear(20, 5),
    # No activation!
    nn.Linear(5, 1)
)

# This is mathematically equivalent to:
model_simple = nn.Linear(10, 1)

# Same power as single layer!
```

**With activation:** Non-linear transformations → complex patterns!

```python
# Network WITH activation functions
model_nonlinear = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),      # ← Non-linearity!
    nn.Linear(20, 5),
    nn.ReLU(),      # ← Non-linearity!
    nn.Linear(5, 1)
)

# This can learn complex patterns!
```

**The difference:**

```yaml
Without activation:
  Layer 1: y = W1x + b1
  Layer 2: z = W2y + b2
         = W2(W1x + b1) + b2
         = W2W1x + W2b1 + b2
         = W3x + b3  ← Still just linear!

With activation:
  Layer 1: y = ReLU(W1x + b1)
  Layer 2: z = ReLU(W2y + b2)
         ← Non-linear! Can learn curves, boundaries, etc.
```

## Common Activation Functions

### ReLU (Most Popular)

```python
import torch

def relu(x):
    return torch.maximum(torch.tensor(0.0), x)

x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
print(relu(x))
# tensor([0., 0., 1., 2.])
```

```yaml
ReLU(x) = max(0, x)

Properties:
  ✓ Fast (simple comparison)
  ✓ No vanishing gradient
  ✓ Creates sparsity
  
Use: Hidden layers
```

### Sigmoid (For Probabilities)

```python
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

x = torch.tensor([-2.0, 0.0, 2.0])
print(sigmoid(x))
# tensor([0.1192, 0.5000, 0.8808])
```

```yaml
σ(x) = 1 / (1 + e⁻ˣ)

Properties:
  ✓ Outputs [0, 1]
  ✓ Smooth
  ✗ Vanishing gradients
  
Use: Binary classification output
```

### Tanh (Zero-Centered)

```python
x = torch.tensor([-1.0, 0.0, 1.0])
print(torch.tanh(x))
# tensor([-0.7616,  0.0000,  0.7616])
```

```yaml
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)

Properties:
  ✓ Outputs [-1, 1]
  ✓ Zero-centered
  ✗ Vanishing gradients
  
Use: RNN cells
```

## Where Activation Goes

**After the linear step, before the next layer:**

```python
import torch
import torch.nn as nn

class SingleNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Step 1: Linear (weighted sum)
        z = self.linear(x)
        
        # Step 2: Activation (non-linearity)
        output = self.activation(z)
        
        return output

# Test
neuron = SingleNeuron()
x = torch.tensor([[1.0, 2.0, 3.0]])
output = neuron(x)
print(output)
```

## Practical Example

```python
import torch
import torch.nn as nn

# Temperature prediction neuron
# Inputs: [humidity, pressure, wind_speed]
weather = torch.tensor([[65.0, 1013.0, 10.0]])

# Create neuron
temp_neuron = nn.Sequential(
    nn.Linear(3, 1),
    nn.ReLU()  # Activation ensures non-negative temperature
)

prediction = temp_neuron(weather)
print(f"Predicted temperature: {prediction.item():.1f}°F")
```

## Choosing the Right Activation

```yaml
Hidden layers:
  Default: ReLU
  Modern: SiLU/GELU
  Classical: Tanh

Output layer (depends on task):
  Binary classification: Sigmoid
  Multi-class: Softmax
  Regression: None (linear)
```

**Example network:**

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),        # Hidden layer activation
    nn.Linear(20, 10),
    nn.ReLU(),        # Hidden layer activation
    nn.Linear(10, 1),
    nn.Sigmoid()      # Output activation for binary classification
)
```

## Key Takeaways

✓ **Activation adds non-linearity:** Makes networks powerful

✓ **Applied after linear step:** Linear → Activation → Next layer

✓ **Different types:** ReLU, Sigmoid, Tanh, etc.

✓ **Choose based on task:** Hidden vs output, type of problem

✓ **Without activation:** Multiple layers = single layer (useless!)

**Quick Reference:**

```python
# After linear transformation
z = linear(x)

# Apply activation
output = activation(z)

# Common activations
torch.relu(z)      # ReLU
torch.sigmoid(z)   # Sigmoid  
torch.tanh(z)      # Tanh
F.silu(z)          # SiLU
F.gelu(z)          # GELU
```

**Remember:** Linear step computes, activation function decides! 🎉
