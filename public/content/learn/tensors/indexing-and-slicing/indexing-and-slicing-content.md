---
hero:
  title: "Indexing and Slicing"
  subtitle: "Accessing and Extracting Tensor Elements"
  tags:
    - "🔢 Tensors"
    - "⏱️ 10 min read"
---

Indexing and slicing let you access and extract specific parts of tensors. Think of it like selecting specific pages from a book or specific rows from a spreadsheet!

## The Basics: Indexing Starts at 0

**Important:** In Python and PyTorch, counting starts at **0**, not 1!

![Basic Indexing](/content/learn/tensors/indexing-and-slicing/basic-indexing.png)

**Example:**

```python
import torch

v = torch.tensor([10, 20, 30, 40, 50])

print(v[0])  # Output: tensor(10)  ← First element
print(v[2])  # Output: tensor(30)  ← Third element
print(v[4])  # Output: tensor(50)  ← Fifth element
```

**Manual breakdown:**

```yaml
v = [10, 20, 30, 40, 50]
     ↑   ↑   ↑   ↑   ↑
    [0] [1] [2] [3] [4]

v[0] → 10
v[1] → 20
v[2] → 30
```

**Key rule:** First element is `[0]`, second is `[1]`, third is `[2]`, and so on!

## Negative Indexing

You can count **backwards from the end** using negative indices:

![Negative Indexing](/content/learn/tensors/indexing-and-slicing/negative-indexing.png)

**Example:**

```python
import torch

v = torch.tensor([10, 20, 30, 40, 50])

print(v[-1])  # Output: tensor(50)  ← Last element
print(v[-2])  # Output: tensor(40)  ← Second from end
print(v[-5])  # Output: tensor(10)  ← Fifth from end (first!)
```

**How it works:**

```yaml
Positive:  [0]  [1]  [2]  [3]  [4]
Values:     10   20   30   40   50
Negative: [-5] [-4] [-3] [-2] [-1]

v[-1] = 50  (last)
v[-2] = 40  (second from last)
v[-3] = 30  (third from last)
```

**Useful trick:** `v[-1]` always gets the last element, no matter the size!

## Matrix Indexing (2D)

For matrices, use `[row, column]`:

![Matrix Indexing](/content/learn/tensors/indexing-and-slicing/matrix-indexing.png)

**Example:**

```python
import torch

A = torch.tensor([[10, 20, 30, 40],
                  [50, 60, 70, 80],
                  [90, 100, 110, 120]])

print(A[0, 0])    # Output: tensor(10)   ← Top-left
print(A[1, 2])    # Output: tensor(70)   ← Row 1, Col 2
print(A[2, 3])    # Output: tensor(120)  ← Bottom-right
print(A[-1, -1])  # Output: tensor(120)  ← Also bottom-right!
```

**Manual breakdown:**

```yaml
         Col 0  Col 1  Col 2  Col 3
Row 0:    10     20     30     40
Row 1:    50     60     70     80
Row 2:    90    100    110    120

A[1, 2] → Row 1, Column 2 → 70
A[0, 3] → Row 0, Column 3 → 40
```

**Pattern:** `[row, column]` always - row first, column second!

## Slicing: Getting Multiple Elements

Slicing uses the syntax `[start:end]` where **end is NOT included**!

![Slicing Basics](/content/learn/tensors/indexing-and-slicing/slicing-basics.png)

**Example:**

```python
import torch

v = torch.tensor([10, 20, 30, 40, 50, 60])

print(v[1:4])    # Output: tensor([20, 30, 40])
print(v[0:3])    # Output: tensor([10, 20, 30])
print(v[3:6])    # Output: tensor([40, 50, 60])
```

**Manual breakdown:**

```yaml
v = [10, 20, 30, 40, 50, 60]
     [0] [1] [2] [3] [4] [5]

v[1:4] gets indices: 1, 2, 3  (stops BEFORE 4)
       →  [20, 30, 40]

v[0:3] gets indices: 0, 1, 2
       →  [10, 20, 30]
```

**Critical:** `v[1:4]` gets elements at positions 1, 2, and 3. It does NOT include position 4!

## Slicing Shortcuts

You can omit start or end:

```python
import torch

v = torch.tensor([10, 20, 30, 40, 50, 60])

print(v[:3])     # Output: tensor([10, 20, 30])  ← From start to 3
print(v[3:])     # Output: tensor([40, 50, 60])  ← From 3 to end
print(v[:])      # Output: tensor([10, 20, 30, 40, 50, 60])  ← Everything!
```

**What they mean:**

```yaml
v[:3]  → v[0:3]  → Start at 0, stop before 3
v[3:]  → v[3:6]  → Start at 3, go to end
v[:]   → v[0:6]  → All elements (copy)
```

## Matrix Slicing

Slicing works in 2D too!

![Matrix Slicing](/content/learn/tensors/indexing-and-slicing/matrix-slicing.png)

**Example:**

```python
import torch

A = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

# Get a sub-matrix
print(A[1:3, 1:3])
# Output:
# tensor([[ 6,  7],
#         [10, 11]])

# Get entire row 2
print(A[2, :])
# Output: tensor([9, 10, 11, 12])

# Get entire column 2
print(A[:, 2])
# Output: tensor([3, 7, 11, 15])
```

**Manual breakdown:**

```yaml
A[1:3, 1:3] means:
- Rows 1 to 3 (not including 3) → rows 1, 2
- Cols 1 to 3 (not including 3) → cols 1, 2

Result:
[[6,  7],
 [10, 11]]

A[2, :] means:
- Row 2
- All columns (:)
→ [9, 10, 11, 12]

A[:, 2] means:
- All rows (:)
- Column 2
→ [3, 7, 11, 15]
```

**Remember:** `:` means "all" (all rows or all columns)

## Step Slicing

Add a **step** to skip elements: `[start:end:step]`

![Step Slicing](/content/learn/tensors/indexing-and-slicing/step-slicing.png)

**Example:**

```python
import torch

v = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70])

print(v[::2])    # Output: tensor([0, 20, 40, 60])  ← Every 2nd
print(v[1::2])   # Output: tensor([10, 30, 50, 70]) ← Start 1, every 2nd
print(v[::3])    # Output: tensor([0, 30, 60])      ← Every 3rd
print(v[::-1])   # Output: tensor([70, 60, 50, 40, 30, 20, 10, 0])  ← Reversed!
```

**How it works:**

```yaml
v[::2]  → Start at 0, take every 2nd element
       → Indices: 0, 2, 4, 6
       → Values: [0, 20, 40, 60]

v[1::3] → Start at 1, take every 3rd element
       → Indices: 1, 4, 7
       → Values: [10, 40, 70]

v[::-1] → Negative step reverses!
       → Values: [70, 60, 50, 40, 30, 20, 10, 0]
```

**Cool trick:** `v[::-1]` reverses any tensor!

## Multiple Elements at Once

You can use lists to select specific indices:

```python
import torch

v = torch.tensor([10, 20, 30, 40, 50])

# Select indices 0, 2, 4
indices = torch.tensor([0, 2, 4])
result = v[indices]

print(result)  # Output: tensor([10, 30, 50])
```

**For matrices:**

```python
import torch

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Get specific rows
rows = torch.tensor([0, 2])
result = A[rows]

print(result)
# Output:
# tensor([[1, 2, 3],
#         [7, 8, 9]])
```

## Practical Example: Batch Processing

```python
import torch

# Batch of 5 samples, each with 3 features
batch = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0],
                      [10.0, 11.0, 12.0],
                      [13.0, 14.0, 15.0]])

# Get first 3 samples
first_three = batch[:3]
print(first_three)
# tensor([[ 1.,  2.,  3.],
#         [ 4.,  5.,  6.],
#         [ 7.,  8.,  9.]])

# Get last 2 samples
last_two = batch[-2:]
print(last_two)
# tensor([[10., 11., 12.],
#         [13., 14., 15.]])

# Get all samples, but only first 2 features
first_two_features = batch[:, :2]
print(first_two_features)
# tensor([[ 1.,  2.],
#         [ 4.,  5.],
#         [ 7.,  8.],
#         [10., 11.],
#         [13., 14.]])
```

**What happened:**

```yaml
batch[:3] → First 3 rows (samples 0, 1, 2)
batch[-2:] → Last 2 rows (samples 3, 4)
batch[:, :2] → All rows, first 2 columns (features 0, 1)
```

## Modifying with Indexing

You can change values using indexing:

```python
import torch

v = torch.tensor([10, 20, 30, 40, 50])

# Change single element
v[2] = 999
print(v)  # tensor([ 10,  20, 999,  40,  50])

# Change slice
v[0:2] = torch.tensor([100, 200])
print(v)  # tensor([100, 200, 999,  40,  50])

# Set all to same value
v[:] = 0
print(v)  # tensor([0, 0, 0, 0, 0])
```

## 3D Indexing

For 3D tensors (like batches of images):

```python
import torch

# 2 batches, 3 rows, 4 columns
tensor_3d = torch.randn(2, 3, 4)

# Get first batch
first_batch = tensor_3d[0]      # Shape: (3, 4)

# Get element from second batch, row 1, col 2
element = tensor_3d[1, 1, 2]    # Single value

# Get all batches, row 0, all columns
slice_3d = tensor_3d[:, 0, :]   # Shape: (2, 4)
```

**Pattern:** `[batch, row, col]` for 3D tensors

## Common Patterns

### Get First/Last Row

```python
A = torch.randn(5, 3)

first_row = A[0]     # or A[0, :]
last_row = A[-1]     # or A[-1, :]
```

### Get First/Last Column

```python
A = torch.randn(5, 3)

first_col = A[:, 0]
last_col = A[:, -1]
```

### Get Main Diagonal

```python
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

diagonal = torch.diag(A)
print(diagonal)  # tensor([1, 5, 9])
```

### Skip Every Other Row

```python
A = torch.randn(10, 3)

every_other_row = A[::2]  # Rows 0, 2, 4, 6, 8
```

## Common Gotchas

### ❌ Gotcha 1: End Index Not Included

```python
v = torch.tensor([10, 20, 30, 40, 50])

# v[1:4] gets indices 1, 2, 3 (NOT 4!)
print(v[1:4])  # tensor([20, 30, 40])

# To include index 4, use v[1:5]
print(v[1:5])  # tensor([20, 30, 40, 50])
```

### ❌ Gotcha 2: Slicing Creates a View

```python
v = torch.tensor([1, 2, 3, 4, 5])
slice_v = v[1:4]

# Modifying slice also modifies original!
slice_v[0] = 999

print(v)        # tensor([  1, 999,   3,   4,   5])
print(slice_v)  # tensor([999,   3,   4])

# Use .clone() for a copy
slice_copy = v[1:4].clone()
slice_copy[0] = 100
print(v)  # tensor([  1, 999,   3,   4,   5])  ← Unchanged!
```

### ❌ Gotcha 3: Integer vs Slice

```python
A = torch.randn(3, 4)

# Integer index reduces dimensions
row = A[0]      # Shape: (4,)  ← 1D tensor

# Slice keeps dimensions
row = A[0:1]    # Shape: (1, 4)  ← Still 2D!
```

## Key Takeaways

✓ **Indexing starts at 0:** First element is `[0]`, not `[1]`

✓ **Negative indexing:** `-1` is last, `-2` is second from last

✓ **Slicing:** `[start:end]` - end is NOT included!

✓ **Colon means all:** `A[:, 2]` = all rows, column 2

✓ **Step:** `[::2]` = every 2nd element, `[::-1]` = reverse

✓ **Views not copies:** Slicing creates views - use `.clone()` for copies

**Quick Reference:**

```python
# Basic indexing
v[0]           # First element
v[-1]          # Last element
A[1, 2]        # Row 1, column 2

# Slicing
v[1:4]         # Elements 1, 2, 3
v[:3]          # First 3 elements
v[3:]          # From index 3 to end
v[:]           # All elements

# 2D slicing
A[1:3, 2:4]    # Rows 1-2, columns 2-3
A[0, :]        # First row
A[:, 0]        # First column

# Step slicing
v[::2]         # Every 2nd element
v[::-1]        # Reversed
```

**Congratulations!** You now know how to access any part of any tensor! This is essential for data processing and neural networks. 🎉
