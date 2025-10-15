---
hero:
  title: "Building a Layer"
  subtitle: "Creating Layers of Neurons"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 8 min read"
---

A layer is a collection of neurons that process inputs together. It's the fundamental unit of neural networks!

![Layer Structure](/content/learn/neural-networks/building-a-layer/layer-structure.png)

## From Single Neuron to Layer

Remember that a single neuron takes multiple inputs and produces one output? A **layer** extends this idea: take multiple inputs and produce multiple outputs, where each output comes from a different neuron.

Think of it like a team where each member (neuron) looks at the same information (inputs) but focuses on different aspects, producing different insights (outputs).

### The Mathematical View

**Single neuron:**
```
y = f(w^T x + b)
One output, one set of weights
```

**Layer of neurons:**
```
Y = f(WX + b)
Multiple outputs, matrix of weights
```

Each row of W represents one neuron's weights!

## What is a Layer?

**Layer = Multiple neurons working in parallel**

Let's see this transition from single to multiple:

```python
import torch.nn as nn

# Single neuron: 1 output
neuron = nn.Linear(10, 1)  # 10 inputs → 1 output
```

This creates ONE neuron with 10 weights (one per input) plus 1 bias.

```python
# Layer of 5 neurons: 5 outputs
layer = nn.Linear(10, 5)   # 10 inputs → 5 outputs
```

This creates FIVE neurons, each with 10 weights plus 1 bias. Total: 5 × (10 + 1) = 55 parameters!

**Key insight:** Each output is computed by a different neuron working on the same inputs.

## Creating Your First Layer

Let's build a layer step by step and understand what's happening:

### Step 1: Create the Layer

```python
import torch
import torch.nn as nn

# Create layer: 3 inputs → 4 outputs
layer = nn.Linear(in_features=3, out_features=4)
```

This creates 4 neurons, each receiving 3 inputs.

### Step 2: Prepare Input Data

```python
# One sample with 3 features
x = torch.tensor([[1.0, 2.0, 3.0]])
```

The double brackets `[[...]]` create shape `(1, 3)` for batch processing.

### Step 3: Run Forward Pass

```python
output = layer(x)

print(output.shape)  # torch.Size([1, 4])
print(output)
# tensor([[0.234, -1.123, 0.567, 2.134]], grad_fn=<AddmmBackward0>)
```

**We got 4 outputs!** One from each neuron.

### Understanding the Computation

Let's see what really happened behind the scenes:

```yaml
4 neurons, each with:
  - 3 weights (one per input feature)
  - 1 bias

Total parameters: 4 × (3 + 1) = 16 parameters
```

**Each neuron computes independently:**
```
Neuron 1: w₁₁×1.0 + w₁₂×2.0 + w₁₃×3.0 + b₁ = 0.234
Neuron 2: w₂₁×1.0 + w₂₂×2.0 + w₂₃×3.0 + b₂ = -1.123
Neuron 3: w₃₁×1.0 + w₃₂×2.0 + w₃₃×3.0 + b₃ = 0.567
Neuron 4: w₄₁×1.0 + w₄₂×2.0 + w₄₃×3.0 + b₄ = 2.134
```

**In matrix form:**
```
Output = XW^T + b
where W is (4, 3) and b is (4,)
```

All 4 neurons process the input simultaneously - that's the power of parallel computation!

## Adding Activation Functions

A layer by itself just performs linear transformations. To make it powerful, we need to add **activation functions**:

### Building a Complete Layer

Let's create a proper layer with both linear transformation and activation:

```python
class LayerWithActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
```

**Setting up the components:**
- Linear layer: Does the weighted sum
- ReLU activation: Adds non-linearity

```python
    def forward(self, x):
        # First: linear transformation
        z = self.linear(x)
        # Then: activation
        output = self.activation(z)
        return output
```

Or more concisely:

```python
    def forward(self, x):
        return self.activation(self.linear(x))
```

### Using the Layer

```python
# Create layer: 10 inputs → 20 outputs
layer = LayerWithActivation(10, 20)

# Process a batch of data
x = torch.randn(32, 10)  # 32 samples, 10 features each
output = layer(x)

print(output.shape)  # torch.Size([32, 20])
```

**What's happening:**
- 32 samples go in
- Each passes through 20 neurons
- We get 32 × 20 matrix out
- All processed in one operation!

## Stacking Layers Together

The real power comes from combining multiple layers. Each layer transforms the data, extracting progressively higher-level features:

### Creating a Multi-Layer Network

```python
# Stack layers together
model = nn.Sequential(
    nn.Linear(784, 256),    # Layer 1
    nn.ReLU(),
    nn.Linear(256, 128),    # Layer 2
    nn.ReLU(),
    nn.Linear(128, 10)      # Output layer
)
```

**The architecture:**
```
Input: 784 features (28×28 image flattened)
  ↓
Layer 1: 784 → 256 (extract 256 patterns)
  ↓ ReLU
Layer 2: 256 → 128 (combine into 128 higher-level patterns)
  ↓ ReLU
Output: 128 → 10 (map to 10 class scores)
```

### Watching Data Transform

Let's trace data through each layer to see the transformations:

```python
x = torch.randn(1, 784)
print(f"Input shape: {x.shape}")  # torch.Size([1, 784])
```

**After first linear layer:**
```python
x = model[0](x)  # First linear: 784 → 256
print(f"After layer 1: {x.shape}")  # torch.Size([1, 256])
```

**After ReLU (shape unchanged):**
```python
x = model[1](x)  # ReLU activation
print(f"After ReLU: {x.shape}")  # torch.Size([1, 256])
# Shape same, but negative values are now zero
```

**After second linear layer:**
```python
x = model[2](x)  # Second linear: 256 → 128
print(f"After layer 2: {x.shape}")  # torch.Size([1, 128])
```

**Final output:**
```python
x = model[3](x)  # ReLU
x = model[4](x)  # Final linear: 128 → 10
print(f"Final output: {x.shape}")  # torch.Size([1, 10])
```

**The transformation journey:**
```
784 features → 256 features → 128 features → 10 class scores
(raw pixels) (low patterns) (high patterns) (digit probabilities)
```

## Building Custom Layers

Sometimes you need more than just linear + activation. Let's build a sophisticated custom layer:

### The Modern Layer Recipe

Modern layers often include multiple components. Let's build one following best practices:

```python
class CustomLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
```

**Each component has a purpose:**
- **Linear**: The core transformation (learnable weights)
- **BatchNorm**: Normalizes activations (faster training)
- **ReLU**: Adds non-linearity
- **Dropout**: Prevents overfitting

### The Processing Pipeline

Now let's see the data flow through each component:

```python
    def forward(self, x):
        # Step 1: Linear transformation
        x = self.linear(x)
        # Shape: (batch, in_dim) → (batch, out_dim)
        
        # Step 2: Normalize
        x = self.norm(x)
        # Standardizes values (mean=0, std=1)
        
        # Step 3: Activate
        x = self.activation(x)
        # Zeros out negatives
        
        # Step 4: Regularize
        x = self.dropout(x)
        # Randomly zeros 20% of values during training
        
        return x
```

**The pipeline:**
```
Input → Linear → Normalize → ReLU → Dropout → Output
```

### Using the Custom Layer

```python
# Create layer: 100 inputs → 50 outputs
layer = CustomLayer(100, 50)

# Process a batch
x = torch.randn(32, 100)  # 32 samples
output = layer(x)

print(output.shape)  # torch.Size([32, 50])
print(f"Non-zero values: {(output != 0).sum().item()}/{output.numel()}")
# Some values are zero due to ReLU and Dropout!
```

**Behind the scenes:**
- Linear: 100 × 50 = 5,000 weights + 50 biases = 5,050 parameters
- BatchNorm: 50 × 2 = 100 parameters (scale and shift)
- Total: 5,150 learnable parameters

## Key Takeaways

✓ **Layer = Multiple neurons:** Process inputs in parallel

✓ **nn.Linear(in, out):** Creates a layer

✓ **Add activation:** After linear transformation

✓ **Stack layers:** Build deep networks

✓ **Custom layers:** Combine multiple operations

**Quick Reference:**

```python
# Basic layer
layer = nn.Linear(input_dim, output_dim)

# Layer with activation
layer = nn.Sequential(
    nn.Linear(in_dim, out_dim),
    nn.ReLU()
)

# Multi-layer network
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

**Remember:** Layers are just multiple neurons working together! 🎉
