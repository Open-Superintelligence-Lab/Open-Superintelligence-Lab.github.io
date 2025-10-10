---
hero:
  title: "Building a Neuron in Python"
  subtitle: "Implementing a Neuron from Scratch"
  tags:
    - "🧠 Neuron"
    - "⏱️ 10 min read"
---

Let's build a complete, working neuron from scratch using pure Python and PyTorch!

![Neuron Code](/content/learn/neuron-from-scratch/building-a-neuron-in-python/neuron-code.png)

## Simple Neuron Class

**Example:**

```python
import torch
import torch.nn as nn

class Neuron(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        # Linear step
        z = self.linear(x)
        
        # Activation
        output = self.activation(z)
        
        return output

# Create neuron with 3 inputs
neuron = Neuron(num_inputs=3)

# Make prediction
x = torch.tensor([[1.0, 2.0, 3.0]])
prediction = neuron(x)

print(prediction)
# tensor([[0.6789]], grad_fn=<SigmoidBackward0>)
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create neuron
neuron = Neuron(num_inputs=2)

# Training data (AND gate)
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])

y = torch.tensor([[0.0],
                  [0.0],
                  [0.0],
                  [1.0]])

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(neuron.parameters(), lr=0.5)

# Training loop
for epoch in range(1000):
    # Forward pass
    predictions = neuron(X)
    
    # Calculate loss
    loss = criterion(predictions, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test the trained neuron
print("\\nTrained neuron predictions:")
with torch.no_grad():
    for i, (input_vals, target_val) in enumerate(zip(X, y)):
        pred = neuron(input_vals.unsqueeze(0))
        print(f"{input_vals.tolist()} → {pred.item():.3f} (target: {target_val.item()})")
```

## From Scratch (No nn.Linear)

Build a neuron with just tensors:

```python
import torch

class ManualNeuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias randomly
        self.weights = torch.randn(num_inputs, requires_grad=True)
        self.bias = torch.randn(1, requires_grad=True)
    
    def forward(self, x):
        # Linear step: w·x + b
        z = torch.dot(self.weights, x) + self.bias
        
        # Activation: sigmoid
        output = 1 / (1 + torch.exp(-z))
        
        return output
    
    def parameters(self):
        return [self.weights, self.bias]

# Create and test
neuron = ManualNeuron(num_inputs=3)
x = torch.tensor([1.0, 2.0, 3.0])
output = neuron.forward(x)

print(output)
# tensor([0.7234], grad_fn=<MulBackward0>)
```

## Training From Scratch

```python
import torch

# Manual neuron (from above)
neuron = ManualNeuron(num_inputs=2)

# Training data
X = torch.tensor([[1.0, 2.0],
                  [2.0, 3.0],
                  [3.0, 4.0]])
y = torch.tensor([0.0, 0.0, 1.0])

learning_rate = 0.1

# Training loop
for epoch in range(100):
    total_loss = 0
    
    for i in range(len(X)):
        # Forward pass
        prediction = neuron.forward(X[i])
        
        # Loss (MSE)
        loss = (prediction - y[i]) ** 2
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Update weights manually
        with torch.no_grad():
            for param in neuron.parameters():
                param -= learning_rate * param.grad
                param.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Test
print("\\nPredictions after training:")
for i in range(len(X)):
    pred = neuron.forward(X[i])
    print(f"Input: {X[i].tolist()}, Prediction: {pred.item():.3f}, Target: {y[i].item()}")
```

## Complete Neuron with All Features

```python
import torch
import torch.nn as nn

class CompleteNeuron(nn.Module):
    def __init__(self, num_inputs, activation='relu'):
        super().__init__()
        self.linear = nn.Linear(num_inputs, 1)
        
        # Choose activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()  # No activation
    
    def forward(self, x):
        z = self.linear(x)
        output = self.activation(z)
        return output
    
    def get_weights(self):
        return self.linear.weight.data
    
    def get_bias(self):
        return self.linear.bias.data

# Create neurons with different activations
relu_neuron = CompleteNeuron(3, activation='relu')
sigmoid_neuron = CompleteNeuron(3, activation='sigmoid')

x = torch.tensor([[1.0, 2.0, 3.0]])

print("ReLU:", relu_neuron(x))
print("Sigmoid:", sigmoid_neuron(x))
```

## Real-World Application

```python
import torch
import torch.nn as nn
import torch.optim as optim

# House price predictor
class HousePriceNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 features: size, bedrooms, age
        self.linear = nn.Linear(3, 1)
        # No activation (regression)
    
    def forward(self, features):
        price = self.linear(features)
        return price

# Training data
houses = torch.tensor([[1500.0, 3.0, 10.0],  # [size, bedrooms, age]
                       [2000.0, 4.0, 5.0],
                       [1200.0, 2.0, 15.0],
                       [1800.0, 3.0, 8.0]])

prices = torch.tensor([[300000.0],  # Actual prices
                       [450000.0],
                       [250000.0],
                       [380000.0]])

# Create and train
model = HousePriceNeuron()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0000001)

# Train
for epoch in range(500):
    predictions = model(houses)
    loss = criterion(predictions, prices)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.2f}")

# Predict new house
new_house = torch.tensor([[1600.0, 3.0, 12.0]])
predicted_price = model(new_house)
print(f"\\nPredicted price: ${predicted_price.item():,.0f}")
```

## Key Takeaways

✓ **Building blocks:** Linear layer + activation function

✓ **From scratch:** Can build with just tensors

✓ **PyTorch way:** Use `nn.Module` and `nn.Linear`

✓ **Training:** Forward → loss → backward → update

✓ **Flexible:** Choose different activations for different tasks

**Quick Reference:**

```python
# Simple neuron
class Neuron(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(self.linear(x))

# Training
model = Neuron(num_inputs=5)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Remember:** You just built a neuron from scratch! This is the foundation of all neural networks! 🎉
