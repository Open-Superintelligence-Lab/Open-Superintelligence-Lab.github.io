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

## What is a Layer?

**Layer = Multiple neurons working in parallel**

```python
import torch.nn as nn

# Single neuron
neuron = nn.Linear(10, 1)  # 10 inputs → 1 output

# Layer of 5 neurons
layer = nn.Linear(10, 5)   # 10 inputs → 5 outputs

# Each output is from a different neuron!
```

## Creating a Layer

```python
import torch
import torch.nn as nn

# Create layer: 3 inputs → 4 outputs
layer = nn.Linear(in_features=3, out_features=4)

# Test
x = torch.tensor([[1.0, 2.0, 3.0]])  # 1 sample, 3 features
output = layer(x)

print(output.shape)  # torch.Size([1, 4])
print(output)
# tensor([[0.234, -1.123, 0.567, 2.134]], grad_fn=<AddmmBackward0>)
# 4 different outputs!
```

**What happened:**

```yaml
4 neurons, each with:
  - 3 weights (one per input)
  - 1 bias

Total parameters: 4×(3+1) = 16 parameters

Each neuron computes:
  neuron1: w1·x + b1
  neuron2: w2·x + b2
  neuron3: w3·x + b3
  neuron4: w4·x + b4
```

## Layer with Activation

```python
class LayerWithActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(self.linear(x))

# Use it
layer = LayerWithActivation(10, 20)
x = torch.randn(32, 10)  # Batch of 32
output = layer(x)

print(output.shape)  # torch.Size([32, 20])
```

## Multiple Layers

```python
# Stack layers together
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    
    nn.Linear(256, 128),
    nn.ReLU(),
    
    nn.Linear(128, 10)
)

# Each layer transforms the data
x = torch.randn(1, 784)
print(x.shape)  # torch.Size([1, 784])

x = model[0](x)  # First linear
print(x.shape)  # torch.Size([1, 256])

x = model[1](x)  # ReLU
print(x.shape)  # torch.Size([1, 256])

x = model[2](x)  # Second linear
print(x.shape)  # torch.Size([1, 128])
```

## Custom Layer

```python
class CustomLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

# Use custom layer
layer = CustomLayer(100, 50)
x = torch.randn(32, 100)
output = layer(x)
print(output.shape)  # torch.Size([32, 50])
```

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
