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

## The Forward Pass

**Forward pass = Input → Linear → Activation → Output**

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
# tensor([[0.8176]]) ← Prediction!
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

## Batch Predictions

Process multiple samples at once:

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

When making predictions (not training), use `torch.no_grad()`:

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
