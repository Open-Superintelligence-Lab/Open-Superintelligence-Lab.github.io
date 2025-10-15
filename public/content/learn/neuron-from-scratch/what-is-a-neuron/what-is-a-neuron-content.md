---
hero:
  title: "What is a Neuron"
  subtitle: "The Basic Building Block of Neural Networks"
  tags:
    - "🧠 Neuron"
    - "⏱️ 8 min read"
---

A neuron is the **fundamental building block** of neural networks. Just like biological neurons in your brain, artificial neurons process inputs and produce outputs!

## The Foundation of Intelligence

Before diving into the mechanics, let's understand why neurons matter. In your brain, billions of neurons work together to:
- Recognize faces
- Understand language
- Make decisions
- Learn from experience

Artificial neurons attempt to mimic this by creating mathematical functions that can learn patterns from data. While much simpler than biological neurons, they're powerful enough to:
- Drive cars autonomously
- Translate between languages
- Beat world champions at chess
- Generate human-like text and images

**The key insight:** Complex intelligence emerges from combining many simple processing units!

## Biological vs Artificial

![Biological vs Artificial](/content/learn/neuron-from-scratch/what-is-a-neuron/biological-vs-artificial.png)

**Biological neuron:**
- Receives signals through dendrites
- Processes in cell body
- Sends output through axon

**Artificial neuron:**
- Receives numerical inputs
- Processes with math (multiply, sum, activate)
- Outputs a single number

**Both:** Transform multiple inputs into one output!

## The Five Parts of a Neuron

![Neuron Parts](/content/learn/neuron-from-scratch/what-is-a-neuron/neuron-parts.png)

### 1. **Inputs** (x₁, x₂, x₃, ...)

The data fed into the neuron:

```python
inputs = [2.0, 3.0, 1.0]
```

**Real examples:**
- Pixel values from an image
- Features of a house (size, bedrooms, age)
- Word embeddings

### 2. **Weights** (w₁, w₂, w₃, ...)

How important each input is:

```python
weights = [0.5, -0.3, 0.8]
```

**What weights mean:**
- Positive weight → input increases output
- Negative weight → input decreases output
- Large |weight| → input is important
- Small weight → input matters less

### 3. **Multiply** (inputs × weights)

Each input gets multiplied by its weight:

```python
products = [2.0 × 0.5,  3.0 × -0.3,  1.0 × 0.8]
         = [1.0,       -0.9,        0.8]
```

### 4. **Sum** (Σ)

Add all products together, plus a bias:

```python
sum_total = 1.0 + (-0.9) + 0.8 + bias
          = 0.9 + 0  # assuming bias = 0
          = 0.9
```

### 5. **Activation Function**

Apply non-linearity (like ReLU, sigmoid, etc.):

```python
output = ReLU(0.9) = 0.9  # Positive, so unchanged
```

## The Complete Formula

**Output = Activation(Σ(weights · inputs) + bias)**

Or in math notation:
**y = f(w₁x₁ + w₂x₂ + w₃x₃ + ... + b)**

### Mathematical Breakdown

Let's understand every symbol in this formula:

**y** - The output (prediction) of the neuron
- This is what the neuron produces
- Could represent: a probability, a class score, a continuous value, etc.

**f** - The activation function
- Adds non-linearity to the model
- Common choices: ReLU, sigmoid, tanh
- Without it, networks can't learn complex patterns

**w₁, w₂, w₃, ...** - The weights (parameters)
- These are the **learnable** parts
- Each input gets its own weight
- During training, these adjust to minimize error
- Can be positive (increase output) or negative (decrease output)

**x₁, x₂, x₃, ...** - The inputs (features)
- The data fed into the neuron
- Examples: pixel values, measurements, embeddings
- These are **fixed** for a given data point

**b** - The bias term
- A learnable offset
- Shifts the decision boundary
- Allows the neuron to activate even when all inputs are zero

**Σ** - The summation symbol
- Means "add up all the products"
- Σ(wᵢxᵢ) = w₁x₁ + w₂x₂ + w₃x₃ + ...

### Why This Formula?

This specific structure comes from trying to model biological neurons mathematically. The weighted sum represents how strongly different inputs should influence the output, and the activation function mimics the "firing" of a biological neuron (it only activates above a certain threshold).

Where:
- `x` = inputs (data/features)
- `w` = weights (learned parameters)
- `b` = bias (learned offset)
- `f` = activation function (non-linearity)

## Simple Example

![Simple Neuron](/content/learn/neuron-from-scratch/what-is-a-neuron/simple-neuron.png)

Let's work through a complete example with actual numbers to see how each part of the neuron works together.

**Example:**

```python
import torch

# Inputs
x = torch.tensor([2.0, 3.0, 1.0])

# Weights
w = torch.tensor([0.5, -0.3, 0.8])

# Bias
b = torch.tensor(0.0)

# Step 1: Multiply
products = x * w
print(products)
# tensor([ 1.0000, -0.9000,  0.8000])

# Step 2: Sum
weighted_sum = products.sum() + b
print(weighted_sum)
# tensor(0.9000)

# Step 3: Activation (ReLU)
output = torch.relu(weighted_sum)
print(output)
# tensor(0.9000)
```

**Manual calculation:**

```yaml
Step 1: Multiply each input by its weight
  2 × 0.5 = 1.0
  3 × -0.3 = -0.9
  1 × 0.8 = 0.8

Step 2: Sum everything + bias
  1.0 + (-0.9) + 0.8 + 0 = 0.9

Step 3: Apply activation (ReLU)
  ReLU(0.9) = max(0, 0.9) = 0.9

Final output: 0.9
```

### Understanding Each Step Mathematically

Let's break down what happened in mathematical terms:

**Step 1: Element-wise multiplication**
```
[x₁, x₂, x₃] ⊙ [w₁, w₂, w₃] = [x₁w₁, x₂w₂, x₃w₃]
[2.0, 3.0, 1.0] ⊙ [0.5, -0.3, 0.8] = [1.0, -0.9, 0.8]
```
The ⊙ symbol means element-wise multiplication (also called Hadamard product).

**Step 2: Summation and bias**
```
z = Σᵢ(xᵢwᵢ) + b
z = (x₁w₁) + (x₂w₂) + (x₃w₃) + b
z = 1.0 + (-0.9) + 0.8 + 0
z = 0.9
```
This is the linear combination of inputs.

**Step 3: Activation**
```
y = f(z) = ReLU(z) = max(0, z)
y = max(0, 0.9) = 0.9
```
Since z is positive, ReLU passes it through unchanged.

**In vector notation:**
```
y = f(w^T x + b)
```
Where w^T x is the dot product (same as Σᵢ(xᵢwᵢ)).

## Why Do We Need Neurons?

### They Learn Patterns

Neurons adjust their weights to recognize patterns. This is where the "learning" in machine learning comes from!

```python
# Neuron learning to detect "cat" in images
# After training:
weights = [0.8,   # whiskers → high weight (important!)
           0.9,   # pointy ears → high weight
           0.1,   # background → low weight (not important)
           -0.5]  # dog features → negative (opposite!)

# When it sees a cat image:
cat_features = [1.0, 1.0, 0.2, 0.0]  # Has whiskers, ears
output = sum(cat_features * weights) + bias
# = 0.8 + 0.9 + 0.02 + 0 = 1.72
# → High output = "Yes, cat!"

# When it sees a dog image:
dog_features = [0.0, 0.0, 0.3, 1.0]  # No whiskers/ears, has dog features
output = sum(dog_features * weights) + bias
# = 0 + 0 + 0.03 + -0.5 = -0.47
# → Low output = "No, not cat"
```

## Single Neuron Can Be Powerful

Even one neuron can solve problems! Despite its simplicity, a single neuron can learn to perform logical operations and make binary decisions.

### The Mathematical Power

A single neuron creates a **linear decision boundary** in the input space. For 2D inputs, this is a line; for 3D inputs, a plane; for higher dimensions, a hyperplane.

**Mathematically:**
```
Decision boundary: w₁x₁ + w₂x₂ + b = 0
```

Points on one side of this boundary get classified one way, points on the other side get classified differently.

**Example: AND gate**

Let's see how a neuron can learn Boolean logic:

```python
import torch

def and_gate(x1, x2):
    """Neuron implementing AND logic"""
    w1, w2 = 1.0, 1.0
    bias = -1.5
    
    # Weighted sum
    z = x1 * w1 + x2 * w2 + bias
    
    # Activation (step function)
    output = 1.0 if z > 0 else 0.0
    return output

# Truth table
print(and_gate(0, 0))  # 0 (False AND False = False)
print(and_gate(0, 1))  # 0 (False AND True = False)
print(and_gate(1, 0))  # 0 (True AND False = False)
print(and_gate(1, 1))  # 1 (True AND True = True)
```

**How it works:**

```yaml
Inputs: (1, 1)
  1×1 + 1×1 + (-1.5) = 0.5 > 0 → Output 1 ✓

Inputs: (0, 1)
  0×1 + 1×1 + (-1.5) = -0.5 < 0 → Output 0 ✓

Inputs: (1, 0)
  1×1 + 0×1 + (-1.5) = -0.5 < 0 → Output 0 ✓

Inputs: (0, 0)
  0×1 + 0×1 + (-1.5) = -1.5 < 0 → Output 0 ✓
```

### Why These Specific Weights?

The magic is in the bias of -1.5. Let's understand the decision boundary:

**Decision boundary equation:**
```
1×x₁ + 1×x₂ - 1.5 = 0
x₁ + x₂ = 1.5
```

This creates a line where the sum of inputs equals 1.5. Since our inputs are either 0 or 1:
- (0, 0): sum = 0 < 1.5 → Output 0 ✓
- (0, 1): sum = 1 < 1.5 → Output 0 ✓
- (1, 0): sum = 1 < 1.5 → Output 0 ✓
- (1, 1): sum = 2 > 1.5 → Output 1 ✓

Perfect AND gate behavior!

### More Logic Gates

```python
# OR gate: x₁ OR x₂
def or_gate(x1, x2):
    w1, w2, bias = 1.0, 1.0, -0.5
    z = x1 * w1 + x2 * w2 + bias
    return 1.0 if z > 0 else 0.0
# Activates if either input is 1

# NOT gate: NOT x₁
def not_gate(x1):
    w1, bias = -1.0, 0.5
    z = x1 * w1 + bias
    return 1.0 if z > 0 else 0.0
# Inverts the input
```

**Key insight:** By choosing different weights and biases, a single neuron can perform different computations!

## Many Neurons = Network

```yaml
Single neuron:
  Limited power
  Can learn simple patterns
  Linear decision boundaries only
  
Multiple neurons:
  Combined power
  Can learn complex patterns
  Each neuron specializes in something
  Non-linear decision boundaries
  
Example: Image classification
  Neuron 1: Detects edges
  Neuron 2: Detects curves
  Neuron 3: Detects textures
  ...
  Together: Recognize objects!
```

### The Power of Combination

**Mathematical principle:** While a single neuron can only create linear decision boundaries, combining multiple neurons with non-linear activations allows us to approximate any function!

This is called the **Universal Approximation Theorem**: A neural network with at least one hidden layer can approximate any continuous function to arbitrary accuracy (given enough neurons).

**Simple example with 2 neurons:**

```python
import torch
import torch.nn as nn

# Single neuron - can't learn XOR
single_neuron = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
# This CANNOT learn XOR (not linearly separable)

# Two neurons - can learn XOR!
two_neurons = nn.Sequential(
    nn.Linear(2, 2),  # 2 neurons in hidden layer
    nn.Sigmoid(),
    nn.Linear(2, 1),  # Combine their outputs
    nn.Sigmoid()
)
# This CAN learn XOR!
```

**Why?** The first layer creates two different linear boundaries, and the second layer combines them to create a non-linear decision boundary.

## Key Takeaways

✓ **Neuron = Processor:** Takes inputs, produces output

✓ **Three operations:** Multiply, Sum, Activate

✓ **Weights are key:** They determine what the neuron learns

✓ **Bias shifts:** Adjusts the threshold

✓ **Activation adds non-linearity:** Makes networks powerful

✓ **Building block:** Many neurons = neural network

**The formula:**

```yaml
Output = Activation(Σ(weights × inputs) + bias)

In code:
  output = activation(torch.sum(weights * inputs) + bias)

Or with linear layer:
  output = activation(nn.Linear(inputs))
```

**Remember:** A neuron is just multiply → sum → activate! Everything else builds on this! 🎉
