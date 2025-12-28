---
hero:
  title: "Transposing Tensors"
  subtitle: "Flipping Dimensions and Axes"
  tags:
    - "🔢 Tensors"
    - "⏱️ 10 min read"
---

Transposing is like **flipping** a tensor - rows become columns, and columns become rows. It's simple but incredibly useful!

## The Basic Idea

**Transpose = Swap rows and columns**

Think of it like rotating a table 90 degrees. The first row becomes the first column, the second row becomes the second column, and so on.

When you transpose a 1D tensor (a vector), it actually stays exactly the same in PyTorch! This is a common point of confusion.

**Example:**

```python
import torch

# 1D vector
v = torch.tensor([1, 2, 3, 4])
print(v.shape)  # torch.Size([4])

# Transpose
v_t = v.T
print(v_t.shape)  # torch.Size([4]) - Still the same!
print(torch.equal(v, v_t))  # True
```

To actually turn a 1D vector into a column vector (2D), you need to reshape it:

```python
# Change to column vector (4 rows, 1 column)
v_col = v.reshape(4, 1) 
print(v_col.shape)  # torch.Size([4, 1])

# Now transposing works as expected
v_row = v_col.T
print(v_row.shape)  # torch.Size([1, 4])
```

**Manual visualization:**

```yaml
1D Vector: [1, 2, 3, 4]  →  Shape: (4,)

Column Vector (2D): [[1],
                     [2],
                     [3],
                     [4]]  →  Shape: (4, 1)

Row Vector (2D):    [[1, 2, 3, 4]] → Shape: (1, 4)
```

## Matrix Transpose

This is where transpose really shines! Rows become columns, and columns become rows:

**Example:**

```python
import torch

# Original matrix: 2 rows, 3 columns
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(A.shape)  # torch.Size([2, 3])

# Transpose: 3 rows, 2 columns
A_T = A.T

print(A_T)
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])

print(A_T.shape)  # torch.Size([3, 2])
```

**Manual calculation:**

```yaml
Original (2×3):
[[1, 2, 3],
 [4, 5, 6]]

Transpose (3×2):
[[1, 4],    ← First column becomes first row
 [2, 5],    ← Second column becomes second row
 [3, 6]]    ← Third column becomes third row
```

## How Elements Move

Here's exactly what happens to each element during transpose:

**The pattern:** Position `[i, j]` → Position `[j, i]`

**Example tracking specific elements:**

```yaml
Original position → Transposed position

[0, 0]: value 1  →  [0, 0]: value 1  (stays in place)
[0, 1]: value 2  →  [1, 0]: value 2  (row 0, col 1 → row 1, col 0)
[0, 2]: value 3  →  [2, 0]: value 3
[1, 0]: value 4  →  [0, 1]: value 4
[1, 1]: value 5  →  [1, 1]: value 5  (stays in place)
[1, 2]: value 6  →  [2, 1]: value 6
```

**Key rule:** Just swap the two indices! `[i, j]` becomes `[j, i]`

## Square Matrix Transpose

Square matrices (same number of rows and columns) have a special property:

**Example:**

```python
import torch

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print(A.shape)  # torch.Size([3, 3])

A_T = A.T
print(A_T)
# tensor([[1, 4, 7],
#         [2, 5, 8],
#         [3, 6, 9]])

print(A_T.shape)  # torch.Size([3, 3])
```

**What happens:**

```yaml
Original:           Transposed:
[[1, 2, 3],        [[1, 4, 7],
 [4, 5, 6],   →     [2, 5, 8],
 [7, 8, 9]]         [3, 6, 9]]

Diagonal (1, 5, 9) stays in place!
Everything else flips across the diagonal.
```

**The diagonal stays put:** Elements where row = column don't move!

## Shape Changes

The shape always flips:

```python
# Examples of shape changes
original_shape = (2, 3)
transposed_shape = (3, 2)

original_shape = (5, 7)
transposed_shape = (7, 5)

original_shape = (4, 4)  # Square
transposed_shape = (4, 4)  # Still square!
```

**Quick reference:**

```yaml
(2, 3) → (3, 2)
(5, 1) → (1, 5)
(10, 20) → (20, 10)
(n, m) → (m, n)  ← General pattern
```

## Why Do We Transpose?

The most common reason: **making shapes compatible for matrix multiplication!**

**Example:**

```python
import torch

A = torch.randn(2, 3)  # Shape: (2, 3)
B = torch.randn(4, 3)  # Shape: (4, 3)

# This WON'T work - shapes (2,3) and (4,3) are incompatible
# result = A @ B  # Error! 3 != 4

# Transpose B to make it work!
B_T = B.T  # Shape: (3, 4)

# Now it works!
result = A @ B_T  # (2, 3) @ (3, 4) -> (2, 4)
print(result.shape)  # torch.Size([2, 4])
```

**Real example with actual values:**

```python
import torch

# Two data samples with 3 features each
X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])  # Shape: (2, 3)

# Weight matrix: 3 inputs, 2 outputs (we want this orientation)
W = torch.tensor([[0.1, 0.2],
                  [0.3, 0.4],
                  [0.5, 0.6]])  # Shape: (3, 2)

# This works!
output = X @ W  # (2, 3) @ (3, 2) = (2, 2)
print(output)
# tensor([[2.2000, 2.8000],
#         [4.9000, 6.4000]])

# But if W was stored transposed...
W_stored = W.T  # Shape: (2, 3)

# We need to transpose it back
output = X @ W_stored.T  # (2, 3) @ (3, 2) = (2, 2)
print(output)  # Same result!
```

## Practical Examples

### Example 1: Computing Dot Products

```python
import torch

# Two vectors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Can't use @ directly on 1D tensors for matrix multiply
# But we can reshape and transpose!

a_col = a.reshape(-1, 1)  # Column vector (3, 1)
b_row = b.reshape(1, -1)  # Row vector (1, 3)

# Outer product
outer = a_col @ b_row  # (3, 1) @ (1, 3) = (3, 3)
print(outer)
# tensor([[ 4,  5,  6],
#         [ 8, 10, 12],
#         [12, 15, 18]])

# Inner product (dot product)
inner = b_row @ a_col  # (1, 3) @ (3, 1) = (1, 1)
print(inner)  # tensor([[32]])
```

### Example 2: Batch Matrix Transpose

```python
import torch

# Batch of 3 matrices, each 2×4
batch = torch.randn(3, 2, 4)

# Transpose last two dimensions
batch_T = batch.transpose(-2, -1)  # Now (3, 4, 2)

print(batch.shape)    # torch.Size([3, 2, 4])
print(batch_T.shape)  # torch.Size([3, 4, 2])

# Each of the 3 matrices got transposed individually!
```

### Example 3: Neural Network Weights

```python
import torch

# In neural networks, weights are often stored transposed
# for computational efficiency

batch_size = 32
input_features = 10
output_features = 5

# Input batch
X = torch.randn(batch_size, input_features)  # (32, 10)

# Weights stored as (input, output) for efficiency
W = torch.randn(input_features, output_features)  # (10, 5)

# Forward pass - works directly!
output = X @ W  # (32, 10) @ (10, 5) = (32, 5) ✓

# If weights were stored as (output, input) instead...
W_alt = torch.randn(output_features, input_features)  # (5, 10)

# Need to transpose
output = X @ W_alt.T  # (32, 10) @ (10, 5) = (32, 5) ✓
```

## Common Gotchas

### ❌ Gotcha 1: 1D Tensors Don't Change Much

```python
v = torch.tensor([1, 2, 3])
v_t = v.T

print(torch.equal(v, v_t))  # True!
# 1D tensors look the same after transpose!
```

To actually change a 1D tensor, reshape it first:

```python
v = torch.tensor([1, 2, 3])
v_col = v.reshape(-1, 1)  # Column vector

print(v.shape)      # torch.Size([3])
print(v_col.shape)  # torch.Size([3, 1])
```

### ❌ Gotcha 2: Transpose Creates a View

```python
A = torch.tensor([[1, 2], [3, 4]])
A_T = A.T

# Modifying A_T also modifies A!
A_T[0, 0] = 999

print(A)
# tensor([[999,   2],
#         [  3,   4]])

# Use .clone() if you want a copy
A_T_copy = A.T.clone()
A_T_copy[0, 0] = 42
# A is unchanged
```

## Key Takeaways

✓ **Transpose swaps rows and columns:** `[i, j]` → `[j, i]`

✓ **Shape flips:** `(m, n)` → `(n, m)`

✓ **Main use:** Making shapes compatible for matrix multiplication

✓ **Diagonal stays:** In square matrices, diagonal elements don't move

✓ **Use `.T`:** Simple and clean syntax in PyTorch

**Quick Reference:**

```python
# Basic transpose
A = torch.tensor([[1, 2, 3], [4, 5, 6]])
A_T = A.T  # Shape: (2,3) → (3,2)

# For 3D+ tensors, specify dimensions
B = torch.randn(5, 10, 20)
B_T = B.transpose(1, 2)  # Swap dimensions 1 and 2
# Shape: (5, 10, 20) → (5, 20, 10)

# Transpose last two dimensions (common in batch operations)
C = torch.randn(8, 4, 6)
C_T = C.transpose(-2, -1)  # or C.transpose(1, 2)
# Shape: (8, 4, 6) → (8, 6, 4)
```

**Remember:** Transposing is just flipping! Rows → Columns, Columns → Rows. That's it! 🎉
