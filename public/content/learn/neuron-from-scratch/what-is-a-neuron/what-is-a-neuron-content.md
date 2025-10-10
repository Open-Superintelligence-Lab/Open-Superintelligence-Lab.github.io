---
hero:
  title: "What is a Neuron"
  subtitle: "The Basic Building Block of Neural Networks"
  tags:
    - "🧠 Neuron"
    - "⏱️ 8 min read"
---

A neuron is the **fundamental building block** of neural networks. Just like biological neurons in your brain, artificial neurons process inputs and produce outputs!

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

Where:
- `x` = inputs
- `w` = weights
- `b` = bias
- `f` = activation function

## Simple Example

![Simple Neuron](/content/learn/neuron-from-scratch/what-is-a-neuron/simple-neuron.png)

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

## Why Do We Need Neurons?

### They Learn Patterns

Neurons adjust their weights to recognize patterns:

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

Even one neuron can solve problems:

**Example: AND gate**

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

## Many Neurons = Network

```yaml
Single neuron:
  Limited power
  Can learn simple patterns
  
Multiple neurons:
  Combined power
  Can learn complex patterns
  Each neuron specializes in something
  
Example: Image classification
  Neuron 1: Detects edges
  Neuron 2: Detects curves
  Neuron 3: Detects textures
  ...
  Together: Recognize objects!
```

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
