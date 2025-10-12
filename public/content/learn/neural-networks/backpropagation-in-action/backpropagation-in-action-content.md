---
hero:
  title: "Backpropagation in Action"
  subtitle: "Seeing Gradients Flow Through Networks"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 8 min read"
---

Let's see backpropagation in action with real examples! We'll watch gradients flow backwards through a network, layer by layer.

## Building Our Test Network

First, let's create a small network we can easily inspect:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 3),    # Input layer: 2 → 3
    nn.ReLU(),           # Activation
    nn.Linear(3, 1)      # Output layer: 3 → 1
)
```

**The architecture:**
```
Input (2) → Hidden (3 neurons) → Output (1)
Total parameters: 2×3 + 3 + 3×1 + 1 = 13
```

### Preparing Data

```python
x = torch.tensor([[1.0, 2.0]])      # One sample, 2 features
y_true = torch.tensor([[5.0]])       # Target value
```

### Forward Pass

```python
y_pred = model(x)
print(f"Prediction: {y_pred.item():.3f}")

loss = (y_pred - y_true) ** 2
print(f"Loss: {loss.item():.3f}")
```

Let's say we get:
```
Prediction: 1.234
Loss: 14.186  (very wrong! Target was 5.0)
```

### Backward Pass: Watching Gradients Flow

Now comes the magic - let's see what backpropagation computes:

```python
# Compute all gradients with one call
loss.backward()

# Inspect each parameter's gradient
for name, param in model.named_parameters():
    print(f"\\n{name}:")
    print(f"  Parameter shape: {param.shape}")
    print(f"  Parameter values:\\n{param.data}")
    print(f"  Gradient values:\\n{param.grad}")
```

**Example output:**
```
0.weight:
  Parameter shape: torch.Size([3, 2])
  Parameter values:
    tensor([[ 0.4521, -0.3214],
            [ 0.1234,  0.6543],
            [-0.2341,  0.8765]])
  Gradient values:
    tensor([[-3.7664, -7.5328],
            [-0.0000, -0.0000],
            [-1.2543, -2.5086]])

0.bias:
  Parameter shape: torch.Size([3])
  Gradient values:
    tensor([-3.7664, -0.0000, -1.2543])

2.weight:
  Parameter shape: torch.Size([1, 3])
  Gradient values:
    tensor([[-2.3456, -0.0000, -0.7812]])
```

### Understanding What We're Seeing

**Layer 2 (output) gradients:**
- Direct path from loss
- Gradient shows how output weights affect error

**Layer 1 (hidden) gradients:**
- Gradient flowed through Layer 2 and ReLU
- Notice some gradients are 0 (from ReLU killing negative values!)

**The zero gradients:**
```
When ReLU input was negative:
  ReLU output = 0
  ReLU gradient = 0
  No gradient flows back!
```

This is the "dying ReLU" phenomenon in action!

## Tracing Gradient Flow Step-by-Step

Let's work through a simpler example to see every calculation:

### Step 1: Setup

```python
import torch

# Create computation with three operations
x = torch.tensor([2.0], requires_grad=True)
```

### Step 2: Forward Pass (Build Computation Graph)

```python
y = x ** 2      # y = x²
z = y + 3       # z = y + 3  
loss = z ** 2   # loss = z²
```

**What happened:**
```
x=2 → [²] → y=4 → [+3] → z=7 → [²] → loss=49
```

PyTorch secretly recorded:
```
Computation graph:
  x → square → y → add(3) → z → square → loss
```

### Step 3: Backward Pass (Apply Chain Rule)

```python
loss.backward()  # Magic!

print(f"x = {x.item()}")               # 2.0
print(f"y = {y.item()}")               # 4.0
print(f"z = {z.item()}")               # 7.0
print(f"loss = {loss.item()}")         # 49.0
print(f"\\ndloss/dx = {x.grad.item()}")  # 56.0
```

### Step 4: Verify with Manual Chain Rule

Let's do the math ourselves:

**Chain rule application:**
```
dloss/dx = (dloss/dz) × (dz/dy) × (dy/dx)

Step 1: dloss/dz = d/dz(z²) = 2z = 2(7) = 14
Step 2: dz/dy = d/dy(y+3) = 1
Step 3: dy/dx = d/dx(x²) = 2x = 2(2) = 4

Final: dloss/dx = 14 × 1 × 4 = 56 ✓
```

PyTorch computed the same thing automatically!

### Visualizing the Gradient Flow

```
Forward computation:
  2.0 → [²] → 4.0 → [+3] → 7.0 → [²] → 49.0

Backward gradients (chain rule):
  56 ← [×4] ← 14 ← [×1] ← 14 ← [×2z=14] ← 1
  ↑                                        ↑
  ∂L/∂x                                  ∂L/∂L=1
```

Each step multiplies by the local gradient!

## Real Training Example with Backpropagation

Let's train a network on actual data and watch it learn:

### The Problem: Learn y = 2x

Let's train a model to learn a simple linear relationship.

**Step 1: Create Model**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple linear model: y = wx + b
model = nn.Linear(1, 1)

# Check initial (random) parameters
print(f"Initial weight: {model.weight.item():.3f}")
print(f"Initial bias: {model.bias.item():.3f}")
```

**Step 2: Prepare Data**

```python
# True relationship: y = 2x
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
```

We know the answer should be: weight=2, bias=0. Can the network learn this?

**Step 3: Setup Training**

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
```

**Step 4: Training Loop**

Let's see gradients in action:

```python
for epoch in range(50):
    # FORWARD PASS
    pred = model(X)
    loss = criterion(pred, y)
    
    # BACKWARD PASS (backpropagation!)
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()         # Compute new gradients
    
    # Peek at gradients (before update)
    if epoch % 10 == 0:
        print(f"\\nEpoch {epoch}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Weight: {model.weight.item():.3f}, Gradient: {model.weight.grad.item():.3f}")
        print(f"  Bias: {model.bias.item():.3f}, Gradient: {model.bias.grad.item():.3f}")
    
    # UPDATE WEIGHTS
    optimizer.step()
```

**Expected output:**
```
Epoch 0:
  Loss: 25.4321
  Weight: 0.523, Gradient: -12.345
  Bias: -0.234, Gradient: -8.765

Epoch 10:
  Loss: 8.2341
  Weight: 1.234, Gradient: -5.678
  Bias: -0.123, Gradient: -3.456

Epoch 20:
  Loss: 2.7650
  Weight: 1.678, Gradient: -2.345
  Bias: -0.045, Gradient: -1.234

Epoch 30:
  Loss: 0.9234
  Weight: 1.876, Gradient: -0.987
  Bias: -0.012, Gradient: -0.456

Epoch 40:
  Loss: 0.3145
  Weight: 1.954, Gradient: -0.321
  Bias: -0.003, Gradient: -0.123
```

See how:
- Loss decreases over time
- Gradients get smaller (approaching minimum)
- Weights approach the true values (2.0 and 0.0)

**Step 5: Check Final Result**

```python
print(f"\\nFinal learned parameters:")
print(f"  Weight: {model.weight.item():.2f}")  # ~2.0
print(f"  Bias: {model.bias.item():.2f}")      # ~0.0
print(f"\\nTrue relationship: y = 2x + 0")
print(f"Learned relationship: y = {model.weight.item():.2f}x + {model.bias.item():.2f}")
```

**Result:**
```
Learned relationship: y = 1.98x + 0.01
```

Almost perfect! The network learned the pattern from data alone!

## Key Takeaways

✓ **Backprop:** Computes gradients efficiently

✓ **Chain rule:** Multiplies gradients backwards

✓ **Automatic:** PyTorch handles it

✓ **Essential:** Makes training possible

**Remember:** Backprop = automatic gradient calculation through layers! 🎉
