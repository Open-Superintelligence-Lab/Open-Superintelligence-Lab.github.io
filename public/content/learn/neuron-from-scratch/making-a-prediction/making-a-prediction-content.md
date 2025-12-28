---
hero:
  title: "Making a Prediction"
  subtitle: "Using a Neuron for Forward Pass"
  tags:
    - "🧠 Neuron"
    - "⏱️ 8 min read"
---

Now that we understand neurons, let's use one to **make predictions**! This is called the **forward pass**.

![Prediction Flow](/content/learn/neuron-from-scratch/making-a-prediction/prediction-flow.png)

## The Mathematical Flow

The forward pass is the computational graph that transforms inputs into predictions. It's called "forward" because data flows in one direction: from input to output.

**Mathematical representation:**
```
Input x → Linear transformation z = Wx + b → Activation y = f(z) → Output
```

Each arrow represents a mathematical operation that can be:
- Computed efficiently (forward pass)
- Differentiated automatically (backward pass for training)

## The Forward Pass

**Forward pass = Input → Linear → Activation → Output**

### Detailed Mathematical Steps

For a single neuron:

**Step 1: Linear combination**
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
  = Σᵢwᵢxᵢ + b
  = w^T x + b
```

**Step 2: Non-linear activation**
```
y = f(z)
```

**Step 3: Output interpretation**
```
For regression: y is the predicted value
For classification: y is the class probability or score
```

**Complete function:**
```
y = f(w^T x + b)
```

This is the fundamental equation of a neuron!

**Example:**

```python
import torch
import torch.nn as nn

# Create a trained neuron (pretend it's already trained)
neuron = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

# Set trained weights manually (normally learned from data)
with torch.no_grad():
    neuron[0].weight = nn.Parameter(torch.tensor([[0.5, 0.8]]))
    neuron[0].bias = nn.Parameter(torch.tensor([-0.3]))

# Make a prediction
input_data = torch.tensor([[1.0, 2.0]])  # New data point
prediction = neuron(input_data)

print(prediction)
# tensor([[0.8581]]) ← Prediction!
```

**Manual calculation:**

```yaml
Input: [1.0, 2.0]
Weights: [0.5, 0.8]
Bias: -0.3

Step 1: Linear
  z = (1.0×0.5) + (2.0×0.8) + (-0.3)
    = 0.5 + 1.6 - 0.3
    = 1.8

Step 2: Activation (Sigmoid)
  output = 1 / (1 + e⁻¹·⁸)
         = 1 / (1 + 0.165)
         = 0.858

Prediction: 0.858 or 85.8% probability
```

### Understanding the Computation Graph

```
x=[1.0, 2.0] ──┐
               ├──→ z=1.8 ────→ y=σ(1.8)=0.858
w=[0.5, 0.8] ──┤
b=-0.3 ────────┘

Forward flow:
1. Inputs and parameters combine linearly
2. Result passes through activation
3. Final output is produced
```

Each operation is:
- **Deterministic**: Same input → same output
- **Differentiable**: Can compute gradients
- **Composable**: Can chain multiple neurons

## Batch Predictions

Process multiple samples at once for computational efficiency!

### Why Batching Matters

**Mathematical motivation:**
- Matrix operations are highly optimized on GPUs
- Processing N samples independently: O(N) separate operations
- Processing N samples as batch: O(1) matrix operation
- Can be 10-100x faster!

**Batch computation:**
```
Single sample: y = f(w^T x + b)          ← Vector operations
Batch:         Y = f(XW + b)              ← Matrix operations

Where X is (batch_size × features) matrix
```

```python
import torch
import torch.nn as nn

neuron = nn.Sequential(
    nn.Linear(3, 1),
    nn.ReLU()
)

# Batch of 5 samples, 3 features each
batch = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 3.0, 4.0],
                      [0.5, 1.0, 1.5],
                      [3.0, 2.0, 1.0],
                      [1.5, 2.5, 3.5]])

# Make predictions for all samples
predictions = neuron(batch)

print(predictions.shape)  # torch.Size([5, 1])
print(predictions)
# tensor([[...],
#         [...],
#         [...],
#         [...],
#         [...]]) ← 5 predictions!
```

## Real-World Example: Binary Classification

```python
import torch
import torch.nn as nn

# Spam detector neuron
class SpamNeuron(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, email_features):
        # Linear step
        logit = self.linear(email_features)
        
        # Activation (probability)
        probability = self.sigmoid(logit)
        
        return probability

# Create and use
spam_detector = SpamNeuron(num_features=100)

# New email features
email = torch.randn(1, 100)

# Predict
spam_probability = spam_detector(email)
print(f"Spam probability: {spam_probability.item():.1%}")

if spam_probability > 0.5:
    print("Prediction: SPAM")
else:
    print("Prediction: NOT SPAM")
```

## Step-by-Step Prediction

```python
import torch

# Input
x = torch.tensor([3.0, 2.0])

# Learned parameters
w = torch.tensor([0.4, 0.6])
b = torch.tensor(0.2)

# Step 1: Weighted sum
print("Inputs:", x)
print("Weights:", w)

products = x * w
print("Products:", products)
# tensor([1.2, 1.2])

weighted_sum = products.sum() + b
print("Sum + bias:", weighted_sum)
# tensor(2.6)

# Step 2: Activation
activated = torch.relu(weighted_sum)
print("After ReLU:", activated)
# tensor(2.6)

# Final prediction
print(f"\\nPrediction: {activated.item()}")
```

**Output:**

```yaml
Inputs: tensor([3., 2.])
Weights: tensor([0.4, 0.6])
Products: tensor([1.2, 1.2])
Sum + bias: tensor(2.6)
After ReLU: tensor(2.6)

Prediction: 2.6
```

## Inference Mode

When making predictions (not training), use `torch.no_grad()` to disable gradient computation.

### Why Disable Gradients?

**During training:**
```
y = f(Wx + b)
↓
PyTorch tracks: What operations were used?
                What were the inputs?
                How to compute gradients?
↓
Stores computation graph in memory
```

**During inference:**
```
y = f(Wx + b)
↓
No tracking needed! Just compute the output.
↓
Saves memory and speeds up computation
```

**Memory savings:**
```
With gradients: Stores full computation graph
Without gradients: Stores only the final output

For large models: Can save GBs of memory!
```

```python
import torch

model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

# For prediction (inference)
with torch.no_grad():
    input_data = torch.randn(1, 10)
    prediction = model(input_data)
    print(prediction)

# Why? Saves memory (doesn't track gradients)
```

## Key Takeaways

✓ **Forward pass:** Input → Linear → Activation → Output

✓ **Batch processing:** Handle multiple samples at once

✓ **Inference mode:** Use `torch.no_grad()` when not training

✓ **Prediction:** Just run the forward pass!

**Quick Reference:**

```python
# Single prediction
output = model(input_data)

# Batch predictions
outputs = model(batch_data)

# Inference mode (no gradients)
with torch.no_grad():
    prediction = model(new_data)
```

**Remember:** Making predictions is just running the forward pass! 🎉
