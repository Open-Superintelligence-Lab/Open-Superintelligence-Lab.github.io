---
hero:
  title: "Creating Special Tensors"
  subtitle: "Zeros, Ones, Identity Matrices and More"
  tags:
    - "🔢 Tensors"
    - "⏱️ 10 min read"
---

Instead of manually typing out every value, PyTorch provides quick ways to create common tensor patterns. These are incredibly useful!

## Why Special Tensors Matter

In deep learning, you rarely create tensors by typing out individual values. Instead, you need tensors initialized in specific patterns:

- **Zeros** for bias initialization and padding
- **Random values** for weight initialization (breaking symmetry)
- **Identity matrices** for linear transformations
- **Sequences** for positional encodings and indexing

Understanding these patterns is crucial because **how you initialize your tensors directly affects your model's ability to learn**. Poor initialization can lead to vanishing/exploding gradients, while good initialization helps your model converge faster!

## Zeros and Ones

The most basic special tensors: filled with all 0s or all 1s.

### The Theory Behind Zeros

Zeros represent the additive identity - adding zero to any number leaves it unchanged. In neural networks, we often initialize bias terms to zero because:
- **No initial preference:** The network starts without favoring any particular direction
- **Weights handle learning:** The weights will learn the important patterns
- **Numerical stability:** Zero is a safe starting point that won't cause explosions

Zeros are also used for **padding** in sequences and images, where we need to fill space without adding information.

![Zeros and Ones](/content/learn/tensors/creating-special-tensors/zeros-ones.png)

### Creating Zeros

**Example:**

```python
import torch

# Create 2×3 matrix of zeros
zeros = torch.zeros(2, 3)

print(zeros)
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

print(zeros.shape)  # torch.Size([2, 3])
```

**More examples:**

```python
# 1D tensor of zeros
torch.zeros(5)
# tensor([0., 0., 0., 0., 0.])

# 3D tensor of zeros
torch.zeros(2, 3, 4)
# tensor([[[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]],
#         [[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]]])
```

### The Theory Behind Ones

Ones represent the multiplicative identity - multiplying by one leaves values unchanged. In neural networks, ones are useful for:
- **Masks:** A matrix of ones means "keep everything" (before applying specific masking)
- **Initialization:** Some layers (like certain normalization layers) start with weights of 1
- **Scaling:** When you want to apply a uniform transformation

While less common than zeros for initialization, ones serve as building blocks for more complex initialization schemes.

### Creating Ones

**Example:**

```python
import torch

# Create 2×3 matrix of ones
ones = torch.ones(2, 3)

print(ones)
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

print(ones.shape)  # torch.Size([2, 3])
```

**When to use:**

```yaml
zeros():
  - Initialize weights to zero
  - Create padding
  - Initialize bias terms
  
ones():
  - Create masks (all True)
  - Initialize certain layers
  - Multiply by constant values
```

## Identity Matrix

An identity matrix has 1s on the diagonal, 0s everywhere else:

![Identity Matrix](/content/learn/tensors/creating-special-tensors/identity-matrix.png)

### The Mathematical Foundation

The identity matrix is one of the most important concepts in linear algebra. It has a special property: **multiplying any matrix by the identity matrix returns the original matrix unchanged**.

Mathematically: `A @ I = A` and `I @ A = A`

Think of it like the number 1 in multiplication - it's the "do nothing" transformation. In neural networks, identity matrices are used for:

- **Residual connections:** Starting with identity helps gradient flow
- **Initialization:** Some architectures initialize certain weights as identity matrices
- **Linear layers:** Understanding identity helps debug whether layers are learning useful transformations

The diagonal structure (1s on diagonal, 0s elsewhere) means each output dimension gets exactly its corresponding input dimension, with no mixing.

**Example:**

```python
import torch

# Create 4×4 identity matrix
identity = torch.eye(4)

print(identity)
# tensor([[1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 0., 1.]])
```

**Properties:**

```yaml
torch.eye(n) creates:
  - n × n square matrix
  - 1s on diagonal (where row = column)
  - 0s everywhere else

Special property:
  A @ eye(n) = A  (multiplying by identity doesn't change A)
```

**More examples:**

```python
# 3×3 identity
I = torch.eye(3)
print(I)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

# Test the property: A @ I = A
A = torch.randn(3, 3)
result = A @ I
print(torch.allclose(A, result))  # True!
```

## Random Tensors

Random tensors are crucial for initializing neural network weights!

![Random Tensors](/content/learn/tensors/creating-special-tensors/random-tensors.png)

### Why Randomness is Essential

**The Symmetry Breaking Problem:** If you initialize all weights to the same value (like all zeros or all ones), every neuron in a layer will compute the exact same thing. During backpropagation, they'll all receive the same gradients and update identically. **Your neurons would never learn different features!**

Random initialization breaks this symmetry - each neuron starts with different weights, so they learn to detect different patterns. This is fundamental to how neural networks work.

**The Distribution Matters:** Not all random initialization is equal! The distribution you choose affects:
- **Gradient flow:** How well gradients propagate through layers
- **Training speed:** How quickly your model converges
- **Final performance:** Whether your model can reach good solutions

Let's look at the different random distributions PyTorch provides:

### torch.rand() - Uniform Distribution

Creates random values **uniformly distributed between 0 and 1**.

**The Theory:** A uniform distribution means every value in the range [0, 1) has an equal probability of being selected. Imagine rolling a continuous dice where every outcome from 0 to 1 is equally likely.

**Why uniform [0, 1)?**
- All values are positive (useful for probabilities)
- Bounded range prevents extreme values
- Equal probability across the range

However, uniform distribution is rarely used for weight initialization because it doesn't naturally maintain variance through layers. It's better suited for other purposes:

```python
import torch

# Random values in [0, 1)
random_uniform = torch.rand(2, 3)

print(random_uniform)
# tensor([[0.2347, 0.8723, 0.4512],
#         [0.6234, 0.1156, 0.9901]])

# All values are between 0 and 1
```

**When to use:**

```yaml
Good for:
  - Dropout masks
  - Random sampling [0, 1)
  - Probabilities
```

### torch.randn() - Normal Distribution

Creates random values from a **normal (Gaussian) distribution** with mean 0 and standard deviation 1.

**The Theory:** The normal distribution (also called Gaussian distribution) is the famous "bell curve". It has special properties:
- **Mean (μ) = 0:** Values are centered around zero (equal probability of positive/negative)
- **Standard deviation (σ) = 1:** Controls the spread (68% of values within ±1)
- **Symmetric:** The curve is symmetric around the mean
- **Rare extremes:** Very large or very small values are rare

**Why normal distribution for weights?**

1. **Zero mean:** Prevents bias toward positive or negative values
2. **Symmetric:** Treats positive and negative updates equally
3. **Small initial values:** Most values are small (close to 0), preventing saturation
4. **Mathematical properties:** Works well with gradient-based optimization
5. **Central Limit Theorem:** As signals pass through layers, they tend toward normal distribution

This is why **torch.randn() is the standard for weight initialization** in neural networks!

```python
import torch

# Random values from normal distribution
random_normal = torch.randn(2, 3)

print(random_normal)
# tensor([[-0.5234,  1.2301, -1.1142],
#         [ 0.0832, -0.7329,  0.4501]])

# Values can be negative or positive
# Most values are close to 0
```

**When to use:**

```yaml
BEST for:
  - Weight initialization (most common!)
  - Adding noise to data
  - Sampling from Gaussian
```

**This is the most common way to initialize neural network weights!**

### torch.randint() - Random Integers

Creates random **integers** in a specified range.

**The Theory:** Unlike continuous distributions (rand, randn), randint produces discrete values. Each integer in the specified range has equal probability of being selected (uniform discrete distribution).

**Common uses:**
- **Class labels:** When generating dummy training data
- **Token IDs:** In NLP, words are represented as integer indices
- **Random sampling:** Selecting random indices for batches
- **Simulation:** Any scenario requiring discrete random choices

```python
import torch

# Random integers from 0 to 9 (10 excluded)
random_ints = torch.randint(0, 10, (2, 3))

print(random_ints)
# tensor([[3, 7, 1],
#         [9, 2, 5]])

# All values are integers between 0 and 9
```

**More examples:**

```python
# Random integers from 1 to 6 (dice roll)
dice = torch.randint(1, 7, (10,))
print(dice)
# tensor([4, 2, 6, 1, 3, 5, 2, 4, 6, 1])

# Random integers for class labels
labels = torch.randint(0, 5, (100,))  # 100 labels, classes 0-4
```

## Range Tensors

Create sequences of numbers automatically!

![Arange and Linspace](/content/learn/tensors/creating-special-tensors/arange-linspace.png)

### Why Sequences Matter

In deep learning, you often need ordered sequences of numbers for:
- **Positional encodings:** Telling the model where in a sequence each element is
- **Time steps:** For recurrent networks or time series
- **Grid coordinates:** For convolutions or attention mechanisms
- **Indexing:** Creating coordinate systems for your data

Two different approaches exist: **fixed steps** (arange) vs **fixed count** (linspace). Understanding when to use each is important!

### torch.arange() - Step by Fixed Amount

Creates a sequence with a fixed step size (like Python's `range`).

**The Theory:** When you know the interval between values you need, use `arange`. It's deterministic: given a step size, the exact values are predetermined. The number of values you get depends on the range and step.

**Mathematical pattern:**
```
values = [start, start+step, start+2*step, ..., start+n*step]
where start+n*step < end
```

**Use when:**
- You need specific spacing (every 0.1, every 5, etc.)
- You're creating indices (0, 1, 2, 3...)
- You know the interval but don't care about exact count

```python
import torch

# From 0 to 10, step by 2 (10 not included!)
seq = torch.arange(0, 10, 2)

print(seq)
# tensor([0, 2, 4, 6, 8])
```

**More examples:**

```python
# Default start is 0, default step is 1
torch.arange(5)
# tensor([0, 1, 2, 3, 4])

# Specify start and end
torch.arange(3, 8)
# tensor([3, 4, 5, 6, 7])

# Use decimals
torch.arange(0, 1, 0.2)
# tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000])
```

**Pattern:**

```yaml
torch.arange(start, end, step)
  - Starts at 'start'
  - Stops BEFORE 'end'
  - Increments by 'step'
```

### torch.linspace() - N Evenly Spaced Values

Creates N values evenly spaced between start and end.

**The Theory:** When you know how many values you need, use `linspace`. It guarantees an exact count of values, and automatically calculates the step size to evenly divide the range.

**Mathematical pattern:**
```
step = (end - start) / (n - 1)
values = [start, start+step, start+2*step, ..., end]
```

Notice that both start AND end are included!

**Use when:**
- You need an exact number of points
- Creating coordinate grids for visualization
- Sampling a function at N points
- The exact spacing is less important than the count

```python
import torch

# 5 values evenly spaced from 0 to 1
seq = torch.linspace(0, 1, 5)

print(seq)
# tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

**More examples:**

```python
# 10 points from -1 to 1
torch.linspace(-1, 1, 10)
# tensor([-1.0000, -0.7778, -0.5556, -0.3333, -0.1111,
#          0.1111,  0.3333,  0.5556,  0.7778,  1.0000])

# Great for creating x-axis for plotting
x = torch.linspace(0, 10, 100)  # 100 points from 0 to 10
```

**Key difference:**

```yaml
arange(0, 10, 2):
  - You specify the STEP (2)
  - Result: [0, 2, 4, 6, 8]
  - End NOT included

linspace(0, 10, 5):
  - You specify the COUNT (5 values)
  - Result: [0.0, 2.5, 5.0, 7.5, 10.0]
  - End IS included!
```

## Creating "Like" Tensors

Create new tensors matching another tensor's shape:

![Like Tensors](/content/learn/tensors/creating-special-tensors/like-tensors.png)

### The Shape Preservation Principle

Often in neural networks, you need to create a new tensor with the same shape as an existing tensor. Rather than manually extracting the shape and passing it, the `_like` functions do this automatically.

**Why this matters:**
- **Dynamic shapes:** Your code works regardless of input size
- **Less error-prone:** No risk of typos in shape specifications
- **Cleaner code:** Intent is clearer when you say "zeros like x"
- **Type preservation:** Also copies dtype and device (CPU/GPU)

This is especially useful when writing layers or functions that need to work with arbitrary input sizes!

**Example:**

```python
import torch

# Original tensor
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# Create zeros with same shape
zeros = torch.zeros_like(x)
print(zeros)
# tensor([[0, 0, 0],
#         [0, 0, 0]])

# Create ones with same shape
ones = torch.ones_like(x)
print(ones)
# tensor([[1, 1, 1],
#         [1, 1, 1]])

# Create random with same shape
random = torch.randn_like(x.float())  # Must be float for randn
print(random.shape)  # torch.Size([2, 3])
```

**When to use:**

```yaml
zeros_like():
  - Reset gradients
  - Create zero-initialized tensors matching input

ones_like():
  - Create masks
  - Initialize to constant

randn_like():
  - Add noise matching shape
  - Initialize weights
```

## Practical Examples

### Example 1: Weight Initialization

This is the most important practical use of special tensors. Let's understand why we initialize weights and biases the way we do:

```python
import torch

# Input dimension: 784 (28×28 image flattened)
# Output dimension: 10 (10 classes)
input_dim = 784
output_dim = 10

# Initialize weights with small random values
weights = torch.randn(input_dim, output_dim) * 0.01

# Initialize bias to zeros
bias = torch.zeros(output_dim)

print(f"Weights shape: {weights.shape}")  # (784, 10)
print(f"Bias shape: {bias.shape}")        # (10,)
```

**Why this initialization?**

- **Random weights:** Break symmetry so each neuron learns different features
- **Normal distribution:** Centered at zero, most values are small
- **Scale by 0.01:** Makes values very small to prevent saturation at start
- **Zero bias:** No initial preference, let the network learn the bias

**Note:** Modern networks use more sophisticated schemes like Xavier or He initialization, but the principle is the same - small random weights, zero bias!

### Example 2: Creating a Mask

Masks are binary (True/False) tensors that tell the model which elements to pay attention to and which to ignore. This is crucial when dealing with variable-length sequences:

```python
import torch

# Data batch
data = torch.randn(5, 10)

# Create mask: first 3 samples are valid, last 2 are padding
mask = torch.zeros(5, dtype=torch.bool)
mask[:3] = True

print(mask)
# tensor([ True,  True,  True, False, False])

# Apply mask
valid_data = data[mask]
print(valid_data.shape)  # torch.Size([3, 10])
```

**Why masks matter:**

In real applications, sequences have different lengths. If you're processing sentences, some might be 10 words, others 50 words. You pad them to the same length for batch processing, but you need masks to tell the model "ignore the padding tokens!"

Without masks, the model would try to learn from meaningless padding, which hurts performance.

### Example 3: Creating Training Data

Let's see how all these special tensors come together in a typical deep learning setup. This example shows the structure of data for training a sequence classification model:

```python
import torch

batch_size = 32
sequence_length = 50
embedding_dim = 128

# Input sequences (random for demo)
inputs = torch.randn(batch_size, sequence_length, embedding_dim)

# Labels (random class indices)
labels = torch.randint(0, 10, (batch_size,))

# Attention mask (all ones = all valid)
attention_mask = torch.ones(batch_size, sequence_length)

print(f"Inputs: {inputs.shape}")           # (32, 50, 128)
print(f"Labels: {labels.shape}")           # (32,)
print(f"Mask: {attention_mask.shape}")     # (32, 50)
```

**Understanding the dimensions:**

- **Batch size (32):** Processing 32 sequences at once (parallel computation)
- **Sequence length (50):** Each sequence has 50 time steps (words/tokens)
- **Embedding dim (128):** Each token is represented by 128 numbers
- **Labels (32):** One class label per sequence (0-9 for 10 classes)
- **Mask (32, 50):** One attention value per token in each sequence

This 3D structure (batch × sequence × features) is fundamental to modern deep learning!

## Full vs Empty

Create tensors without initializing values (faster but contains garbage).

### The Performance Trade-off

When you create a tensor with `zeros()` or `ones()`, PyTorch must:
1. Allocate memory
2. Initialize every element to 0 or 1

The `empty()` function skips step 2 - it only allocates memory, leaving whatever values were already there (garbage). This is slightly faster, but you must be careful!

**Use empty() only when:**
- You'll immediately overwrite ALL values anyway
- You're in a performance-critical loop
- You know what you're doing (debugging garbage values is painful!)

For `full()`, you can specify any constant value, making it more flexible than `ones()` or `zeros()`:

```python
import torch

# Create empty tensor (uninitialized - garbage values)
empty = torch.empty(2, 3)
print(empty)
# tensor([[3.6893e+19, 1.5414e-19, 3.0818e-41],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00]])
# Random garbage values!

# Create full tensor (fill with specific value)
sevens = torch.full((2, 3), 7)
print(sevens)
# tensor([[7, 7, 7],
#         [7, 7, 7]])
```

**When to use empty:**

```yaml
torch.empty():
  - When you'll immediately overwrite all values
  - Slightly faster than zeros/ones
  - WARNING: Contains random garbage!
  
torch.full():
  - Fill with any constant value
  - Like ones() but more flexible
```

## Key Takeaways

✓ **zeros() and ones():** All 0s or all 1s

✓ **eye():** Identity matrix (diagonal 1s)

✓ **rand():** Random [0, 1) uniform

✓ **randn():** Random normal distribution (best for weights!)

✓ **randint():** Random integers

✓ **arange():** Sequence with step (end excluded)

✓ **linspace():** N evenly spaced values (end included)

✓ **_like():** Match another tensor's shape

**Quick Reference:**

```python
# Zeros and ones
torch.zeros(3, 4)              # 3×4 matrix of zeros
torch.ones(2, 5)               # 2×5 matrix of ones

# Identity
torch.eye(5)                   # 5×5 identity matrix

# Random
torch.rand(3, 3)               # Uniform [0, 1)
torch.randn(3, 3)              # Normal (μ=0, σ=1)
torch.randint(0, 10, (3, 3))   # Random integers [0, 10)

# Sequences
torch.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)        # [0.00, 0.25, 0.50, 0.75, 1.00]

# Like another tensor
x = torch.randn(2, 3)
torch.zeros_like(x)            # Zeros with shape (2, 3)
torch.ones_like(x)             # Ones with shape (2, 3)
torch.randn_like(x)            # Random with shape (2, 3)

# Fill with value
torch.full((2, 3), 7)          # All 7s
```

**Remember:** Use `torch.randn()` for weight initialization - it's the standard! 🎉
