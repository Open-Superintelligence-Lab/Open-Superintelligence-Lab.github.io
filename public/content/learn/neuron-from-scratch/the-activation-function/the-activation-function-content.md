---
hero:
  title: "The Activation Function"
  subtitle: "Adding Non-Linearity to Neurons"
  tags:
    - "🧠 Neuron"
    - "⏱️ 8 min read"
---

The activation function is what makes neural networks **powerful**. Without it, you'd just have fancy linear regression!

![Activation Comparison](/content/learn/neuron-from-scratch/the-activation-function/activation-comparison.png)

## The Mathematical Necessity

Activation functions introduce **non-linearity** into neural networks. This is not just a design choice - it's a mathematical necessity for learning complex patterns.

### The Universal Approximation Theorem

This theorem states that a neural network with:
- At least one hidden layer
- Non-linear activation functions
- Sufficient neurons

Can approximate any continuous function to arbitrary precision!

**Without non-linearity,** this theorem doesn't hold. The network reduces to simple linear regression, no matter how many layers you stack.

## Why We Need Activation Functions

**Without activation:** No matter how many layers, it's still just linear!

### Mathematical Proof of Collapse

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
  Layer 1: y = σ(W1x + b1)  ← Non-linear σ!
  Layer 2: z = σ(W2y + b2)  ← Non-linear σ!
         ← Can't simplify! Truly complex function!
```

### Detailed Mathematical Collapse

Let's prove why stacking linear layers without activation is pointless:

```
Layer 1: z₁ = W₁x + b₁
Layer 2: z₂ = W₂z₁ + b₂
       = W₂(W₁x + b₁) + b₂
       = W₂W₁x + W₂b₁ + b₂
       
Let W₃ = W₂W₁ and b₃ = W₂b₁ + b₂

Then: z₂ = W₃x + b₃

This is identical to a single layer!
```

**Generalization:** For n linear layers:
```
f(x) = Wₙ(...(W₂(W₁x + b₁) + b₂)...) + bₙ
     = W_combined x + b_combined
```
Always reduces to a single linear transformation!

## Common Activation Functions

Each activation function has unique mathematical properties that make it suitable for different tasks.

### ReLU (Most Popular)

**Mathematical definition:**
```
ReLU(x) = max(0, x) = {
  x   if x > 0
  0   if x ≤ 0
}
```

```python
import torch

def relu(x):
    return torch.maximum(torch.tensor(0.0), x)

x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
print(relu(x))
# tensor([0., 0., 1., 2.])
```

**Derivative (important for backpropagation):**
```
d/dx ReLU(x) = {
  1   if x > 0
  0   if x ≤ 0
  undefined at x = 0 (we use 0 in practice)
}
```

**Properties:**

```yaml
ReLU(x) = max(0, x)

Mathematical properties:
  ✓ Piecewise linear (two linear pieces)
  ✓ Non-saturating for x > 0 (gradient doesn't vanish)
  ✓ Sparse activation (outputs exactly 0 for x ≤ 0)
  ✓ Scale-invariant: ReLU(αx) = α·ReLU(x) for α > 0
  
Computational properties:
  ✓ Extremely fast (one comparison, no exponentials)
  ✓ Gradient is either 0 or 1 (no multiplication needed)
  
Issues:
  ✗ "Dying ReLU": Neurons can get stuck outputting 0
  ✗ Not zero-centered (all outputs ≥ 0)
  ✗ Not differentiable at 0 (but works fine in practice)
  
Use: Hidden layers (default choice)
```

**Why it works:** The simple thresholding at 0 creates a piecewise linear function. When combined across many neurons, these create complex non-linear boundaries!

### Sigmoid (For Probabilities)

**Mathematical definition:**
```
σ(x) = 1/(1 + e^(-x)) = e^x/(e^x + 1)

Alternative form: σ(x) = (1 + tanh(x/2))/2
```

```python
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

x = torch.tensor([-2.0, 0.0, 2.0])
print(sigmoid(x))
# tensor([0.1192, 0.5000, 0.8808])
```

**Derivative:**
```
d/dx σ(x) = σ(x)(1 - σ(x))
```
Beautiful property: the derivative is expressed in terms of the function itself!

**Properties:**

```yaml
σ(x) = 1 / (1 + e^(-x))

Mathematical properties:
  ✓ Smooth and differentiable everywhere
  ✓ Outputs in range (0, 1) - interpretable as probability!
  ✓ Monotonically increasing
  ✓ Symmetry: σ(-x) = 1 - σ(x)
  ✓ S-shaped curve (sigmoid means "S-shaped")
  
Limits:
  lim(x→∞) σ(x) = 1
  lim(x→-∞) σ(x) = 0
  σ(0) = 0.5
  
Issues:
  ✗ Vanishing gradient problem: for |x| > 4, gradient ≈ 0
  ✗ Not zero-centered (outputs always positive)
  ✗ Expensive to compute (exponential)
  
Use: Binary classification output layer only
```

**Gradient vanishing explained:**
```
When x = 10:  σ(10) ≈ 0.9999,  σ'(10) ≈ 0.00005  ← Nearly zero!
When x = -10: σ(-10) ≈ 0.0001, σ'(-10) ≈ 0.00005  ← Nearly zero!
```

### Tanh (Zero-Centered)

**Mathematical definition:**
```
tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)) = 2σ(2x) - 1

Relationship to sigmoid: tanh(x) = 2σ(2x) - 1
```

```python
x = torch.tensor([-1.0, 0.0, 1.0])
print(torch.tanh(x))
# tensor([-0.7616,  0.0000,  0.7616])
```

**Derivative:**
```
d/dx tanh(x) = 1 - tanh²(x) = sech²(x)
```

**Properties:**

```yaml
tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

Mathematical properties:
  ✓ Smooth and differentiable everywhere
  ✓ Output range: (-1, 1) - zero-centered!
  ✓ Monotonically increasing
  ✓ Odd function: tanh(-x) = -tanh(x)
  ✓ Stronger gradients than sigmoid (centered at 0)
  
Limits:
  lim(x→∞) tanh(x) = 1
  lim(x→-∞) tanh(x) = -1
  tanh(0) = 0
  
Issues:
  ✗ Still has vanishing gradient for |x| > 3
  ✗ Expensive to compute (two exponentials)
  
Use: RNN cells, hidden layers (less common now)
```

**Comparison with sigmoid:**
```
tanh(x) = 2σ(2x) - 1

tanh is just a scaled and shifted sigmoid!
Zero-centered version of sigmoid.
```

### Modern Activation Functions

**SiLU (Swish):**
```
SiLU(x) = x · σ(x) = x/(1 + e^(-x))

Properties:
  ✓ Smooth
  ✓ Non-monotonic (small dip at x < 0)
  ✓ Self-gated (x gates itself)
```

**GELU (Gaussian Error Linear Unit):**
```
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

Or: GELU(x) = x·Φ(x) where Φ is standard normal CDF

Properties:
  ✓ Smooth
  ✓ Used in BERT, GPT
  ✓ Stochastic regularization effect
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
