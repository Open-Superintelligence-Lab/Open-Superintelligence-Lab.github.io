---
hero:
  title: "Reshaping Tensors"
  subtitle: "Changing Tensor Dimensions"
  tags:
    - "🔢 Tensors"
    - "⏱️ 10 min read"
---

Reshaping lets you change how data is organized **without changing the actual values**. Same data, different shape!

## The Basic Idea

Reshaping reorganizes elements into a new structure. Think of it like rearranging books on shelves - same books, different arrangement!

![Basic Reshape](/content/learn/tensors/reshaping-tensors/basic-reshape.png)

**Example:**

```python
import torch

# 1D tensor with 6 elements
v = torch.tensor([1, 2, 3, 4, 5, 6])
print(v.shape)  # torch.Size([6])

# Reshape to 2D: 2 rows, 3 columns
matrix = v.reshape(2, 3)
print(matrix)
# tensor([[1, 2, 3],
#         [4, 5, 6]])
print(matrix.shape)  # torch.Size([2, 3])
```

**What happened:**

```yaml
Original:  [1, 2, 3, 4, 5, 6]  → Shape: (6,)

Reshaped:  [[1, 2, 3],
            [4, 5, 6]]         → Shape: (2, 3)

Same 6 elements, new organization!
```

## The Golden Rule

**Total number of elements must stay the same!**

```yaml
6 elements can become:
✓ (6,)      - 1D with 6 elements
✓ (2, 3)    - 2×3 = 6 elements
✓ (3, 2)    - 3×2 = 6 elements
✓ (1, 6)    - 1×6 = 6 elements
✗ (2, 4)    - 2×4 = 8 elements (ERROR!)
```

## Common Reshape Patterns

### Pattern 1: 1D → 2D

```python
import torch

v = torch.tensor([1, 2, 3, 4, 5, 6])

# Make it 2×3
matrix = v.reshape(2, 3)
print(matrix)
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# Make it 3×2
matrix = v.reshape(3, 2)
print(matrix)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])
```

### Pattern 2: 2D → Different 2D

```python
import torch

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # Shape: (2, 3)

B = A.reshape(3, 2)
print(B)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])  # Shape: (3, 2)
```

## Flattening: Any Dimension → 1D

Flattening converts any tensor into a single row:

![Flatten Visual](/content/learn/tensors/reshaping-tensors/flatten-visual.png)

**Example:**

```python
import torch

matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

# Method 1: flatten()
flat = matrix.flatten()
print(flat)  # tensor([1, 2, 3, 4, 5, 6])

# Method 2: reshape(-1)
flat = matrix.reshape(-1)
print(flat)  # tensor([1, 2, 3, 4, 5, 6])

# Method 3: view(-1)
flat = matrix.view(-1)
print(flat)  # tensor([1, 2, 3, 4, 5, 6])
```

**How it reads:**

```yaml
Matrix:
[[1, 2, 3],
 [4, 5, 6]]

Flattens row by row:
Row 0: [1, 2, 3]
Row 1: [4, 5, 6]

Result: [1, 2, 3, 4, 5, 6]
```

## Using -1: Automatic Dimension

Use `-1` to let PyTorch figure out one dimension automatically!

![Auto Dimension](/content/learn/tensors/reshaping-tensors/auto-dimension.png)

**Example:**

```python
import torch

t = torch.arange(12)  # [0, 1, 2, ..., 11] - 12 elements

# You specify columns, PyTorch figures out rows
print(t.reshape(-1, 3))  # (?, 3) → (4, 3)
# tensor([[ 0,  1,  2],
#         [ 3,  4,  5],
#         [ 6,  7,  8],
#         [ 9, 10, 11]])

# You specify rows, PyTorch figures out columns
print(t.reshape(3, -1))  # (3, ?) → (3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# Just -1 means flatten
print(t.reshape(-1))  # (12,)
```

**How it works:**

```yaml
12 elements, reshape(-1, 3):
→ 12 ÷ 3 = 4 rows
→ Result: (4, 3)

12 elements, reshape(2, -1):
→ 12 ÷ 2 = 6 columns
→ Result: (2, 6)
```

**Important:** Only ONE -1 allowed per reshape!

## Squeeze & Unsqueeze

### Unsqueeze: Add a Dimension

```python
import torch

v = torch.tensor([1, 2, 3])  # Shape: (3,)

# Add dimension at position 0
v_unsqueezed = v.unsqueeze(0)
print(v_unsqueezed.shape)  # torch.Size([1, 3])
print(v_unsqueezed)
# tensor([[1, 2, 3]])

# Add dimension at position 1
v_unsqueezed = v.unsqueeze(1)
print(v_unsqueezed.shape)  # torch.Size([3, 1])
print(v_unsqueezed)
# tensor([[1],
#         [2],
#         [3]])
```

### Squeeze: Remove Dimensions of Size 1

```python
import torch

t = torch.tensor([[[1, 2, 3]]])  # Shape: (1, 1, 3)

# Remove all size-1 dimensions
squeezed = t.squeeze()
print(squeezed.shape)  # torch.Size([3])
print(squeezed)  # tensor([1, 2, 3])

# Remove specific dimension
t2 = torch.randn(1, 5, 1, 3)  # Shape: (1, 5, 1, 3)
squeezed = t2.squeeze(0)  # Remove dimension 0
print(squeezed.shape)  # torch.Size([5, 1, 3])
```

**When to use:**

```yaml
Unsqueeze: When you need to match shapes for operations
  (3,) + unsqueeze(1) → (3, 1) for broadcasting

Squeeze: When you want to remove extra dimensions
  (1, 5, 1) + squeeze() → (5,) cleaner shape
```

## Reshape vs View

Both change shape, but there's a difference:

```python
import torch

t = torch.tensor([[1, 2], [3, 4]])

# reshape() - always works, may copy data
r = t.reshape(4)  # Works!

# view() - faster but requires contiguous memory
v = t.view(4)     # Works if contiguous!
```

**Key difference:**

```yaml
.reshape():
  - Always works
  - May create a copy if needed
  - Safer choice

.view():
  - Faster (no copy)
  - Only works on contiguous tensors
  - May fail with error
```

**When to use which:**
- Use `.reshape()` by default (safer)
- Use `.view()` if you know tensor is contiguous and want speed

## Practical Example: Batch Processing

![Batch Reshape](/content/learn/tensors/reshaping-tensors/batch-reshape.png)

```python
import torch

# 3 images, each 2×2 pixels
images = torch.tensor([[[1, 2], [3, 4]],
                       [[5, 6], [7, 8]],
                       [[9, 10], [11, 12]]])

print(images.shape)  # torch.Size([3, 2, 2])

# Flatten each image for neural network
batch = images.reshape(3, -1)
print(batch)
# tensor([[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]])

print(batch.shape)  # torch.Size([3, 4])
# 3 samples, 4 features each - ready for neural network!
```

**What happened:**

```yaml
Original: (3, 2, 2)
  - 3 images
  - Each image is 2×2

Reshaped: (3, 4)
  - 3 samples
  - Each sample has 4 features (flattened image)
```

## Reshaping Rules

![Reshape Rules](/content/learn/tensors/reshaping-tensors/reshape-rules.png)

### ✓ Valid Reshapes

```python
# 12 elements can be reshaped many ways
t = torch.arange(12)  # 12 elements

t.reshape(3, 4)    # ✓ 3×4 = 12
t.reshape(2, 6)    # ✓ 2×6 = 12
t.reshape(1, 12)   # ✓ 1×12 = 12
t.reshape(2, 2, 3) # ✓ 2×2×3 = 12
```

### ✗ Invalid Reshapes

```python
t = torch.arange(12)  # 12 elements

# t.reshape(3, 5)  # ✗ 3×5 = 15 ≠ 12 - ERROR!
# t.reshape(2, 7)  # ✗ 2×7 = 14 ≠ 12 - ERROR!
```

## Real-World Examples

### Example 1: Preparing Data for Linear Layer

```python
import torch

# Batch of 32 images, each 28×28 pixels
images = torch.randn(32, 28, 28)

# Flatten for fully connected layer
flattened = images.reshape(32, -1)
print(flattened.shape)  # torch.Size([32, 784])
# 32 samples, 784 features (28×28)

# Now ready for: output = linear_layer(flattened)
```

### Example 2: Converting Model Output

```python
import torch

# Model outputs 100 predictions, need 10×10 grid
predictions = torch.randn(100)

# Reshape to grid
grid = predictions.reshape(10, 10)
print(grid.shape)  # torch.Size([10, 10])
```

### Example 3: Adding Batch Dimension

```python
import torch

# Single sample
sample = torch.randn(28, 28)
print(sample.shape)  # torch.Size([28, 28])

# Add batch dimension for model
batched = sample.unsqueeze(0)
print(batched.shape)  # torch.Size([1, 28, 28])
# Now it looks like a batch of 1 sample
```

## Common Patterns

### Pattern: Flatten Batch

```python
batch = torch.randn(32, 3, 224, 224)  # 32 images, 3 channels, 224×224
flat = batch.reshape(32, -1)           # (32, 150528)
```

### Pattern: Split into Batches

```python
data = torch.arange(100)
batches = data.reshape(10, 10)  # 10 batches of 10 samples
```

### Pattern: Match Dimensions for Broadcasting

```python
a = torch.randn(5, 3)      # (5, 3)
b = torch.randn(3)         # (3,)

# Add dimension to b for broadcasting
b = b.unsqueeze(0)         # (1, 3)
result = a + b             # Works! (5, 3) + (1, 3)
```

## Common Gotchas

### ❌ Gotcha 1: Element Count Mismatch

```python
t = torch.arange(12)  # 12 elements

# This will ERROR!
# t.reshape(3, 5)  # 15 ≠ 12
```

### ❌ Gotcha 2: Too Many -1

```python
t = torch.arange(12)

# This will ERROR!
# t.reshape(-1, -1)  # Can't infer both dimensions!
```

### ❌ Gotcha 3: View on Non-Contiguous Tensor

```python
t = torch.randn(3, 4)
t_t = t.T  # Transpose makes it non-contiguous

# This might ERROR!
# v = t_t.view(12)

# Use reshape instead:
r = t_t.reshape(12)  # Works!
```

## Key Takeaways

✓ **Same data, new shape:** Reshaping reorganizes elements without changing values

✓ **Element count must match:** Total elements before = total elements after

✓ **Use -1 for auto:** Let PyTorch figure out one dimension

✓ **Flatten with reshape(-1):** Any tensor → 1D

✓ **Unsqueeze adds, squeeze removes:** Manage dimensions of size 1

✓ **reshape() is safer:** Use reshape() by default, view() for speed

**Quick Reference:**

```python
# Basic reshape
t.reshape(2, 3)      # Specific shape
t.reshape(-1, 3)     # Auto rows, 3 columns
t.reshape(-1)        # Flatten to 1D

# Flatten
t.flatten()          # Always returns 1D
t.reshape(-1)        # Also flattens
t.view(-1)           # Flatten (if contiguous)

# Add/remove dimensions
t.unsqueeze(0)       # Add dimension at position 0
t.unsqueeze(1)       # Add dimension at position 1
t.squeeze()          # Remove all size-1 dimensions
t.squeeze(0)         # Remove specific dimension

# Alternative (view is faster but less safe)
t.view(2, 3)         # Like reshape, but needs contiguous tensor
```

**Remember:** Reshaping doesn't change the data, only how it's organized! 🎉
