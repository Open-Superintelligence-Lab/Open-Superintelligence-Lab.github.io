---
hero:
  title: "Creating Special Tensors"
  subtitle: "Zeros, Ones, Identity Matrices and More"
  tags:
    - "🔢 Tensors"
    - "⏱️ 10 min read"
---

Instead of manually typing out every value, PyTorch provides quick ways to create common tensor patterns. These are incredibly useful!

## Zeros and Ones

The most basic special tensors: filled with all 0s or all 1s.

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

### torch.rand() - Uniform Distribution

Creates random values **uniformly distributed between 0 and 1**:

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

Creates random values from a **normal (Gaussian) distribution** with mean 0 and standard deviation 1:

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

Creates random **integers** in a specified range:

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

### torch.arange() - Step by Fixed Amount

Creates a sequence with a fixed step size (like Python's `range`):

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

Creates N values evenly spaced between start and end:

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

### Example 2: Creating a Mask

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

### Example 3: Creating Training Data

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

## Full vs Empty

Create tensors without initializing values (faster but contains garbage):

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
