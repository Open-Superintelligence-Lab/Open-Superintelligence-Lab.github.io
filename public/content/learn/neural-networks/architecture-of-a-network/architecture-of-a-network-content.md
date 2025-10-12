---
hero:
  title: "Architecture of a Network"
  subtitle: "Understanding Neural Network Structure and Design"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 10 min read"
---

A neural network's **architecture** is its structure - how many layers, how many neurons, and how they connect!

![Network Layers](/content/learn/neural-networks/architecture-of-a-network/network-layers.png)

## What is Network Architecture?

Think of architecture as the **blueprint** for your neural network. Just like building a house, you need to decide:
- How many floors (layers)?
- How large is each floor (neurons per layer)?
- How do floors connect (connections)?

These choices dramatically affect what your network can learn and how well it performs!

### Why Architecture Matters

**Different architectures for different tasks:**
- **Shallow & wide**: Good for simple patterns, fast training
- **Deep & narrow**: Better for complex hierarchical patterns
- **Very deep**: Can learn extremely complex functions (like GPT)

The right architecture can mean the difference between 60% and 95% accuracy!

## The Three Essential Parts

**Every neural network has three fundamental components:**

1. **Input Layer:** Receives the raw data
   - Size determined by your data
   - Example: 784 for 28×28 images

2. **Hidden Layers:** Extract features and patterns
   - You design these!
   - Can have 1 to 1000+ hidden layers

3. **Output Layer:** Makes the final prediction
   - Size determined by your task
   - Example: 10 neurons for 10-class classification

**A typical architecture flow:**
```yaml
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
   (784)         (128)              (64)             (10)
```

Notice how the dimensions reduce: 784 → 128 → 64 → 10. This "funnel" pattern compresses information into high-level predictions!

## Building Your First Architecture

Let's build a network architecture step-by-step, understanding each design decision:

### Design Decision 1: Input Size

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input layer: 784 features
        # Why 784? Because 28×28 pixel images = 784 pixels
        self.fc1 = nn.Linear(784, 128)
```

**Input size is determined by your data**, not a choice you make!

### Design Decision 2: First Hidden Layer

```python
        # Hidden layer 1: 784 → 128
        # Why 128? Trade-off between:
        # - Capacity to learn (more neurons = more patterns)
        # - Computational cost (fewer neurons = faster)
```

We chose 128 as a reasonable middle ground. You could try 64, 256, or 512!

### Design Decision 3: Second Hidden Layer

```python
        # Hidden layer 2: 128 → 64
        # Funnel pattern: gradually reduce dimensions
        self.fc2 = nn.Linear(128, 64)
```

**Why reduce from 128 to 64?**
- Compresses information
- Forces network to learn most important features
- Creates "bottleneck" effect

### Design Decision 4: Output Size

```python
        # Output layer: 64 → 10
        # Why 10? We're classifying 10 digit classes (0-9)
        self.fc3 = nn.Linear(64, 10)
```

**Output size matches your task**: 10 classes = 10 outputs, 2 classes = 2 outputs, regression = 1 output.

### The Forward Pass

Now let's define how data flows through our architecture:

```python
    def forward(self, x):
        # Layer 1 with ReLU activation
        x = torch.relu(self.fc1(x))
        
        # Layer 2 with ReLU activation
        x = torch.relu(self.fc2(x))
        
        # Output layer (no activation - we want raw scores)
        x = self.fc3(x)
        
        return x
```

**Why no activation on output?**
- For classification: CrossEntropyLoss includes softmax
- For regression: We want unbounded values

### Creating and Inspecting the Model

```python
model = SimpleNet()
print(model)

# Output shows the architecture:
# SimpleNet(
#   (fc1): Linear(in_features=784, out_features=128, bias=True)
#   (fc2): Linear(in_features=128, out_features=64, bias=True)
#   (fc3): Linear(in_features=64, out_features=10, bias=True)
# )
```

This summary shows all layers and their shapes!

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

## Choosing Layer Sizes: The Art and Science

Choosing the right layer sizes is part art, part science. Let's explore the principles:

### Input Layer: No Choice Here!

```yaml
Input size = Number of features in your data

Examples:
  - 28×28 grayscale image: 784 neurons
  - 32×32 RGB image: 32×32×3 = 3,072 neurons
  - Text with 50 words, 300-dim embeddings: 50×300 = 15,000
  - Tabular data with 20 columns: 20 neurons
```

You don't choose this - your data does!

### Hidden Layers: Your Creative Canvas

This is where you design! Several strategies work:

**Strategy 1: Funnel (most common)**
```
Wide → Narrow → Narrower

Example: 784 → 512 → 256 → 128 → 10
```
Gradually compress information, extracting essential features.

**Strategy 2: Uniform (modern transformers)**
```
Same → Same → Same

Example: 512 → 512 → 512 → 512 → 10
```
Maintain capacity throughout, let the network figure out compression.

**Strategy 3: Bottleneck (autoencoders)**
```
Wide → Narrow → Wide

Example: 784 → 128 → 32 → 128 → 784
```
Force compression in the middle, then reconstruct.

### Output Layer: Task-Dependent

```yaml
Classification (multi-class):
  Size = number of classes
  Example: 10 classes → 10 neurons (one score per class)

Binary classification:
  Size = 1 neuron
  Example: spam/not spam → 1 neuron (probability)

Regression:
  Size = number of values to predict
  Example: house price → 1 neuron
  Example: (x, y) coordinates → 2 neurons
```

## Architecture Patterns in Practice

Let's see common patterns with real code examples:

### Pattern 1: The Funnel (Feature Extraction)

Best for classification tasks where you want to extract key features:

```python
model = nn.Sequential(
    nn.Linear(784, 512),    # Start wide - capture many patterns
    nn.ReLU(),
    nn.Linear(512, 256),    # Compress - combine patterns
    nn.ReLU(),
    nn.Linear(256, 10)      # Final compression - class scores
)
```

**What each layer learns:**
```
Layer 1 (512): Basic patterns (edges, textures)
Layer 2 (256): Mid-level features (shapes, parts)
Output (10): High-level concepts (digit identities)
```

### Pattern 2: Uniform Width (Transformers)

Used in modern architectures like BERT and GPT:

```python
model = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
```

**Why keep same size?**
- Each layer has equal capacity
- No forced compression
- Let the network decide what to learn
- Easier to scale (add/remove layers)

### Pattern 3: Bottleneck (Autoencoders)

Used for dimensionality reduction and feature learning:

```python
model = nn.Sequential(
    # Encoder: compress
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 32),     # Bottleneck - compressed representation!
    nn.ReLU(),
    
    # Decoder: reconstruct
    nn.Linear(32, 128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 784)     # Reconstruct original
)
```

**The bottleneck forces learning:**
- Original: 784 dimensions
- Compressed: 32 dimensions (96% compression!)
- Reconstructed: 784 dimensions

The 32-dimensional bottleneck must capture all important information!

## The Depth vs Width Trade-off

Two ways to add capacity to your network: go **deeper** (more layers) or **wider** (more neurons per layer). Each has different properties!

### What is Depth and Width?

**Depth** = Number of layers (vertical stacking)
**Width** = Neurons per layer (horizontal expansion)

### Experiment 1: Deep and Narrow

```python
# 5 layers, only 20 neurons each
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
)
```

**Character traits:**
- Total params: 10×20 + 20×20 + 20×20 + 20×20 + 20×1 ≈ 1,200
- Depth: 5 hidden layers
- Can learn hierarchical representations

### Experiment 2: Shallow and Wide

```python
# 1 layer, 1000 neurons
shallow_wide = nn.Sequential(
    nn.Linear(10, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1)
)
```

**Character traits:**
- Total params: 10×1000 + 1000×1 = 11,000 (much more!)
- Depth: 1 hidden layer
- More brute force, less structured

### The Mathematical Difference

**Deep networks** learn compositional functions:
```
f(x) = f₅(f₄(f₃(f₂(f₁(x)))))

Each layer builds on previous:
f₁: edges → f₂: shapes → f₃: parts → f₄: objects → f₅: concepts
```

**Wide networks** learn in one shot:
```
f(x) = f₁(x)

One layer tries to learn everything at once
```

### Which Should You Choose?

```yaml
Go DEEP when:
  ✓ Need hierarchical feature learning
  ✓ Complex structured data (images, text)
  ✓ Have good initialization & normalization
  ✓ Examples: ResNet (152 layers), GPT (96 layers)

Go WIDE when:
  ✓ Simple tabular data
  ✓ Want faster training
  ✓ Limited depth works fine
  ✓ Examples: Simple classification, regression

Modern trend: BOTH (wide AND deep)
  - ResNet: deep with wide residual blocks
  - Transformers: deep with wide attention

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

## Understanding Parameter Count

The number of parameters in your network determines its capacity to learn. Let's understand how to count them:

### Manual Parameter Calculation

For a linear layer: **parameters = (input_size × output_size) + output_size**

The first term is weights, the second is biases.

**Example:**
```
Linear(10, 20):
  Weights: 10 × 20 = 200
  Biases: 20
  Total: 220 parameters
```

### Calculating Full Network Parameters

Let's count step-by-step:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),  # ?
    nn.ReLU(),          # ?
    nn.Linear(20, 5)    # ?
)
```

**Layer 1:**
```
Linear(10, 20):
  10 × 20 (weights) + 20 (biases) = 220 parameters
```

**Activation:**
```
ReLU():
  No learnable parameters = 0
```

**Layer 2:**
```
Linear(20, 5):
  20 × 5 (weights) + 5 (biases) = 105 parameters
```

**Total: 220 + 0 + 105 = 325 parameters**

### Automatic Counting in PyTorch

```python
# PyTorch can count for you
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# Output: 325
```

**Why this matters:**
- More parameters = more capacity but harder to train
- Modern models: GPT-3 has 175 billion parameters!
- Your laptop: Probably good up to ~100 million

## Real-World Architecture: MNIST Classifier

Let's build a complete, practical architecture for digit classification:

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
