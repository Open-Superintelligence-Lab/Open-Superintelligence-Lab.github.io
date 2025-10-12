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

## From Theory to Code

We've learned the mathematics of a neuron:
```
y = σ(w·x + b)

Where:
  x = input vector
  w = weights vector
  b = bias scalar
  σ = activation function
```

Now let's implement this in Python! We'll build it **three ways**:
1. **Using PyTorch's building blocks** (practical)
2. **From scratch with tensors** (educational)
3. **Production-ready with features** (real-world)

## Approach 1: Using PyTorch's nn.Module

PyTorch provides `nn.Linear` and `nn.Module` to make building neurons easy.

### Step 1: Understanding nn.Module

```python
import torch
import torch.nn as nn
```

**What is `nn.Module`?**
- Base class for all neural network components
- Handles parameter tracking automatically
- Manages gradient computation
- Provides training/eval modes

**Why inherit from it?**
```python
class Neuron(nn.Module):  # Inherit from nn.Module
    def __init__(self, num_inputs):
        super().__init__()  # Initialize parent class
```

This gives us:
- Automatic parameter registration
- `.parameters()` method
- `.to(device)` for GPU support
- `.train()` and `.eval()` modes

### Step 2: Define the Linear Layer

```python
    def __init__(self, num_inputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, 1)
```

**What `nn.Linear(num_inputs, 1)` creates:**
```
Weight matrix: W ∈ ℝ^(1 × num_inputs)
Bias: b ∈ ℝ^1

Example with num_inputs=3:
  W = [[w₁, w₂, w₃]]  (1×3 matrix)
  b = [b₁]            (1 value)

Computation: z = xW^T + b
```

**Initial values:** Random (Xavier/Kaiming initialization by default)

### Step 3: Choose Activation Function

```python
        self.activation = nn.Sigmoid()
```

**Common activations in PyTorch:**
```python
nn.ReLU()      # f(x) = max(0, x)
nn.Sigmoid()   # f(x) = 1/(1+e^(-x))
nn.Tanh()      # f(x) = tanh(x)
nn.LeakyReLU() # f(x) = max(0.01x, x)
```

### Step 4: Implement Forward Pass

```python
    def forward(self, x):
        # Step 1: Linear transformation
        z = self.linear(x)
        
        # Step 2: Activation
        output = self.activation(z)
        
        return output
```

**What happens:**
```
x = [1.0, 2.0, 3.0]  (input)
  ↓ linear layer
z = w₁×1.0 + w₂×2.0 + w₃×3.0 + b
  = [2.5]  (example)
  ↓ sigmoid
output = 1/(1+e^(-2.5)) = 0.924
```

### Step 5: Complete Neuron Class

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
```

### Testing Your Neuron

```python
# Create neuron with 3 inputs
neuron = Neuron(num_inputs=3)

# Make prediction
x = torch.tensor([[1.0, 2.0, 3.0]])  # Shape: (1, 3) - batch of 1
prediction = neuron(x)

print(f"Input: {x}")
print(f"Prediction: {prediction}")
print(f"Shape: {prediction.shape}")  # torch.Size([1, 1])

# Inspect parameters
print(f"\nWeights: {neuron.linear.weight}")  # Shape: (1, 3)
print(f"Bias: {neuron.linear.bias}")          # Shape: (1,)
```

**Example output:**
```
Input: tensor([[1., 2., 3.]])
Prediction: tensor([[0.6789]], grad_fn=<SigmoidBackward0>)
Shape: torch.Size([1, 1])

Weights: Parameter containing:
tensor([[ 0.2341, -0.1234,  0.4567]], requires_grad=True)
Bias: Parameter containing:
tensor([0.1234], requires_grad=True)
```

**Notice `grad_fn`?** PyTorch is tracking gradients for backpropagation!

## Training a Neuron: Complete Example

Let's train a neuron to learn the AND gate logic.

### The AND Gate Problem

```yaml
Truth table:
  Input 1  Input 2  →  Output
    0        0           0
    0        1           0
    1        0           0
    1        1           1

Only outputs 1 when BOTH inputs are 1
```

### Step 1: Prepare Data

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
                  [1.0, 1.0]])  # Shape: (4, 2)

y = torch.tensor([[0.0],
                  [0.0],
                  [0.0],
                  [1.0]])  # Shape: (4, 1)
```

**Dataset:**
```
4 examples (rows)
2 features per example (columns)
1 target per example
```

### Step 2: Define Loss and Optimizer

```python
# Loss function: Binary Cross Entropy
criterion = nn.BCELoss()

# Optimizer: Stochastic Gradient Descent
optimizer = optim.SGD(neuron.parameters(), lr=0.5)
```

**Why BCE Loss?**
```
Binary classification (0 or 1)
BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

Penalizes wrong predictions heavily
```

**Why SGD?**
```
Simple, effective optimizer
lr=0.5: Learning rate (step size)
Higher lr = faster but less stable
```

### Step 3: Training Loop Breakdown

```python
# Training loop
for epoch in range(1000):
```

**Epoch:** One complete pass through all training data.

```python
    # Forward pass: Make predictions
    predictions = neuron(X)
    # Shape: (4, 1) - one prediction per example
```

**What's happening:**
```
predictions[0] = σ(w₁×0 + w₂×0 + b) = σ(b)
predictions[1] = σ(w₁×0 + w₂×1 + b) = σ(w₂ + b)
predictions[2] = σ(w₁×1 + w₂×0 + b) = σ(w₁ + b)
predictions[3] = σ(w₁×1 + w₂×1 + b) = σ(w₁ + w₂ + b)
```

```python
    # Calculate loss: How wrong are we?
    loss = criterion(predictions, y)
```

**Loss calculation:**
```
Compare predictions to targets
Higher loss = worse performance
Goal: Minimize loss
```

```python
    # Backward pass: Compute gradients
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Compute new gradients
```

**Gradient computation:**
```
∂L/∂w₁ = how much loss changes with w₁
∂L/∂w₂ = how much loss changes with w₂
∂L/∂b  = how much loss changes with b

Automatic via backpropagation!
```

```python
    # Update weights: Take a step
    optimizer.step()
```

**Weight update:**
```
w₁ ← w₁ - lr × ∂L/∂w₁
w₂ ← w₂ - lr × ∂L/∂w₂
b  ← b  - lr × ∂L/∂b

Move in direction that reduces loss
```

```python
    # Log progress
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Step 4: Complete Training Code

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
print("\nTrained neuron predictions:")
with torch.no_grad():  # Disable gradient tracking for inference
    for i, (input_vals, target_val) in enumerate(zip(X, y)):
        pred = neuron(input_vals.unsqueeze(0))
        print(f"{input_vals.tolist()} → {pred.item():.3f} (target: {target_val.item()})")
```

**Expected output:**
```
Epoch 0, Loss: 0.7234
Epoch 200, Loss: 0.3456
Epoch 400, Loss: 0.1234
Epoch 600, Loss: 0.0567
Epoch 800, Loss: 0.0234

Trained neuron predictions:
[0.0, 0.0] → 0.023 (target: 0.0)  ✓
[0.0, 1.0] → 0.045 (target: 0.0)  ✓
[1.0, 0.0] → 0.067 (target: 0.0)  ✓
[1.0, 1.0] → 0.934 (target: 1.0)  ✓

Neuron learned AND gate!
```

## Approach 2: From Scratch (Pure Tensors)

Let's build a neuron **without** `nn.Linear` - just tensors and math!

### Step 1: Manual Initialization

```python
import torch

class ManualNeuron:
    def __init__(self, num_inputs):
        # Initialize weights randomly
        self.weights = torch.randn(num_inputs, requires_grad=True)
        
        # Initialize bias randomly
        self.bias = torch.randn(1, requires_grad=True)
```

**What `requires_grad=True` does:**
```
Tells PyTorch: "Track operations on this tensor"
Enables automatic gradient computation
Essential for training!
```

**Random initialization:**
```
weights: Sample from N(0, 1)
bias: Sample from N(0, 1)

Example with num_inputs=3:
  weights = [0.5234, -1.2341, 0.8234]
  bias = [0.1234]
```

### Step 2: Implement Forward Pass

```python
    def forward(self, x):
        # Linear step: w·x + b
        z = torch.dot(self.weights, x) + self.bias
```

**Dot product manually:**
```
x = [x₁, x₂, x₃]
w = [w₁, w₂, w₃]

z = w₁×x₁ + w₂×x₂ + w₃×x₃ + b

Example:
  x = [1.0, 2.0, 3.0]
  w = [0.5, -0.2, 0.3]
  b = 0.1
  
  z = 0.5×1.0 + (-0.2)×2.0 + 0.3×3.0 + 0.1
    = 0.5 - 0.4 + 0.9 + 0.1
    = 1.1
```

```python
        # Activation: sigmoid function
        output = 1 / (1 + torch.exp(-z))
        
        return output
```

**Manual sigmoid:**
```
σ(z) = 1 / (1 + e^(-z))

For z = 1.1:
  e^(-1.1) = 0.333
  1 + 0.333 = 1.333
  1 / 1.333 = 0.750
  
Output: 0.750
```

### Step 3: Parameter Access

```python
    def parameters(self):
        """Return list of trainable parameters"""
        return [self.weights, self.bias]
```

**Why needed?**
```
Optimizer needs access to parameters
To update them during training
Similar to nn.Module.parameters()
```

### Complete Manual Neuron

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

print(f"Input: {x}")
print(f"Weights: {neuron.weights}")
print(f"Bias: {neuron.bias}")
print(f"Output: {output}")
print(f"Has gradients: {output.requires_grad}")  # True!
```

### Training From Scratch

```python
import torch

# Create manual neuron
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
        
        # Manual weight update
        with torch.no_grad():  # Disable gradient tracking temporarily
            for param in neuron.parameters():
                # Gradient descent: param = param - lr × gradient
                param -= learning_rate * param.grad
                
                # Zero gradients for next iteration
                param.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Test
print("\nPredictions after training:")
with torch.no_grad():
    for i in range(len(X)):
        pred = neuron.forward(X[i])
        print(f"Input: {X[i].tolist()}, Prediction: {pred.item():.3f}, Target: {y[i].item()}")
```

**What we learned by building from scratch:**
- How `nn.Linear` works internally
- What `requires_grad` does
- Manual gradient descent implementation
- How optimizers update parameters

## Approach 3: Production-Ready Neuron

Let's build a flexible, feature-rich neuron for real applications:

### Features to Add

```yaml
1. Configurable activation functions
2. Parameter inspection methods
3. Weight initialization strategies
4. Multiple output support
5. Dropout for regularization
```

### Implementation

```python
import torch
import torch.nn as nn

class CompleteNeuron(nn.Module):
    def __init__(self, num_inputs, activation='relu', dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(num_inputs, 1)
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        else:
            self.activation = nn.Identity()  # No activation
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        z = self.linear(x)
        z = self.dropout(z)  # Apply dropout
        output = self.activation(z)
        return output
    
    def get_weights(self):
        """Return weight values"""
        return self.linear.weight.data
    
    def get_bias(self):
        """Return bias value"""
        return self.linear.bias.data
    
    def set_weights(self, weights):
        """Manually set weights"""
        with torch.no_grad():
            self.linear.weight.copy_(weights)
    
    def set_bias(self, bias):
        """Manually set bias"""
        with torch.no_grad():
            self.linear.bias.copy_(bias)

# Create neurons with different configurations
relu_neuron = CompleteNeuron(3, activation='relu', dropout=0.2)
sigmoid_neuron = CompleteNeuron(3, activation='sigmoid')
tanh_neuron = CompleteNeuron(3, activation='tanh')

x = torch.tensor([[1.0, 2.0, 3.0]])

print("ReLU:", relu_neuron(x))
print("Sigmoid:", sigmoid_neuron(x))
print("Tanh:", tanh_neuron(x))

# Inspect parameters
print("\nWeights:", relu_neuron.get_weights())
print("Bias:", relu_neuron.get_bias())
```

## Real-World Application: House Price Predictor

Let's build a practical neuron for a real problem!

### The Problem

```
Predict house price based on:
  - Size (sq ft)
  - Number of bedrooms
  - Age (years)
```

### Step 1: Design the Neuron

```python
import torch
import torch.nn as nn
import torch.optim as optim

class HousePriceNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 features: size, bedrooms, age
        self.linear = nn.Linear(3, 1)
        # No activation (regression problem)
    
    def forward(self, features):
        price = self.linear(features)
        return price
```

**Why no activation?**
```
Regression: Predict continuous value
Want output: Any real number (price can be any value)
Activation like ReLU would limit range
```

### Step 2: Prepare Data

```python
# Training data
houses = torch.tensor([[1500.0, 3.0, 10.0],  # [size, bedrooms, age]
                       [2000.0, 4.0, 5.0],
                       [1200.0, 2.0, 15.0],
                       [1800.0, 3.0, 8.0]])

prices = torch.tensor([[300000.0],  # Actual prices
                       [450000.0],
                       [250000.0],
                       [380000.0]])
```

**Data normalization (optional but recommended):**
```python
# Normalize features to similar scales
houses_mean = houses.mean(dim=0)
houses_std = houses.std(dim=0)
houses_normalized = (houses - houses_mean) / houses_std

# Normalize prices
prices_mean = prices.mean()
prices_std = prices.std()
prices_normalized = (prices - prices_mean) / prices_std
```

### Step 3: Train the Model

```python
# Create and train
model = HousePriceNeuron()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.SGD(model.parameters(), lr=0.0000001)

# Train
for epoch in range(500):
    predictions = model(houses)
    loss = criterion(predictions, prices)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: ${loss.item():,.0f}")

# Predict new house
new_house = torch.tensor([[1600.0, 3.0, 12.0]])
predicted_price = model(new_house)
print(f"\nPredicted price: ${predicted_price.item():,.0f}")

# Inspect learned parameters
print(f"\nLearned weights: {model.linear.weight.data}")
print(f"Learned bias: {model.linear.bias.data}")
```

**Interpretation:**
```
If weights = [150.0, 50000.0, -5000.0]:
  - Each sq ft adds $150
  - Each bedroom adds $50,000
  - Each year of age reduces $5,000
  
Price = 150×size + 50000×bedrooms - 5000×age + bias
```

## Key Takeaways

✓ **Three approaches:** PyTorch modules, pure tensors, production-ready

✓ **Building blocks:** Linear layer + activation function

✓ **Training cycle:** Forward → loss → backward → update

✓ **From scratch:** Understanding what PyTorch does internally

✓ **Flexible:** Adapt for classification or regression

## Quick Reference

**Simple neuron:**
```python
class Neuron(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(self.linear(x))
```

**Training:**
```python
model = Neuron(num_inputs=5)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(epochs):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Remember:** You just built a neuron from scratch - the foundation of all neural networks! 🎉
