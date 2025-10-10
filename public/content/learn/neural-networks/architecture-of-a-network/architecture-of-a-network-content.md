---
hero:
  title: "Architecture of a Network"
  subtitle: "Understanding Neural Network Structure and Design"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 10 min read"
---

A neural network's **architecture** is its structure - how many layers, how many neurons, and how they connect!

## Basic Architecture

**Typical neural network has three parts:**

1. **Input Layer:** Receives the data
2. **Hidden Layers:** Process and transform
3. **Output Layer:** Makes the prediction

```yaml
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
   (784)         (128)              (64)             (10)
```

## Example Architecture

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input layer → Hidden layer 1
        self.fc1 = nn.Linear(784, 128)
        
        # Hidden layer 1 → Hidden layer 2
        self.fc2 = nn.Linear(128, 64)
        
        # Hidden layer 2 → Output layer
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        # Layer 1
        x = torch.relu(self.fc1(x))
        
        # Layer 2
        x = torch.relu(self.fc2(x))
        
        # Output layer (no activation for logits)
        x = self.fc3(x)
        
        return x

model = SimpleNet()
print(model)
```

**Architecture diagram:**

```yaml
Input: 784 features (28×28 image flattened)
  ↓
Linear(784 → 128) + ReLU
  ↓
Linear(128 → 64) + ReLU
  ↓
Linear(64 → 10) [logits for 10 classes]
  ↓
Output: 10 class scores
```

## Layer Sizes

**How to choose layer sizes:**

```yaml
Input layer:
  Size = number of features
  Example: 28×28 image = 784

Hidden layers:
  Start wide, gradually narrow
  Common pattern: 512 → 256 → 128
  Or: Stay same size

Output layer:
  Size = number of outputs
  Classification: number of classes
  Regression: usually 1
```

**Example patterns:**

```python
# Pattern 1: Funnel (wide to narrow)
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Pattern 2: Uniform (same size)
model = nn.Sequential(
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

# Pattern 3: Bottleneck (narrow middle)
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 32),   # Bottleneck
    nn.ReLU(),
    nn.Linear(32, 128),
    nn.ReLU(),
    nn.Linear(128, 784)
)
```

## Depth vs Width

**Depth = number of layers**
**Width = neurons per layer**

```python
# Deep but narrow
deep_narrow = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)  # 5 layers, 20 neurons each

# Shallow but wide
shallow_wide = nn.Sequential(
    nn.Linear(10, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1)
)  # 2 layers, 1000 neurons
```

**Trade-offs:**

```yaml
Deep networks:
  ✓ Learn hierarchical features
  ✓ More expressive
  ✗ Harder to train
  ✗ Gradient problems

Wide networks:
  ✓ More parameters per layer
  ✓ Easier to train
  ✗ Less feature hierarchy
  ✗ More memory
```

## Common Architectures

### Fully Connected (Dense)

```python
# Every neuron connects to every neuron in next layer
fc_net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

### Convolutional (CNN)

```python
# For images
cnn = nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64*6*6, 10)
)
```

## Counting Parameters

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),  # 10×20 + 20 = 220 params
    nn.ReLU(),          # 0 params
    nn.Linear(20, 5)    # 20×5 + 5 = 105 params
)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# Output: 325
```

## Practical Example: MNIST Classifier

```python
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Input: 28×28 = 784
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output: 10 classes (digits 0-9)
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        # Flatten image
        x = x.view(-1, 784)
        # Forward pass
        return self.network(x)

model = MNISTNet()

# Count parameters
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")
```

## Key Takeaways

✓ **Three parts:** Input → Hidden → Output

✓ **Layer sizes:** Input (features), Hidden (variable), Output (targets)

✓ **Depth:** Number of layers

✓ **Width:** Neurons per layer

✓ **More layers:** More complex patterns

✓ **Design choice:** Many valid architectures

**Quick Reference:**

```python
# Basic architecture template
model = nn.Sequential(
    nn.Linear(input_size, hidden1_size),
    nn.ReLU(),
    nn.Linear(hidden1_size, hidden2_size),
    nn.ReLU(),
    nn.Linear(hidden2_size, output_size)
)

# Count parameters
total = sum(p.numel() for p in model.parameters())
```

**Remember:** Architecture is like a blueprint - it defines your network's structure! 🎉
