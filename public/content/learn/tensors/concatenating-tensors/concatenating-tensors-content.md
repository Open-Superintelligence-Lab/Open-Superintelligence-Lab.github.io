---
hero:
  title: "Concatenating Tensors"
  subtitle: "Combining Multiple Tensors"
  tags:
    - "🔢 Tensors"
    - "⏱️ 9 min read"
---

Concatenation lets you **join multiple tensors together** along a specific dimension. Think of it like gluing pieces together!

## The Basic Idea

**Concatenation = Joining tensors end-to-end along one dimension**

You can join tensors:
- **Vertically** (stack rows on top of each other)
- **Horizontally** (place side by side)
- **Along any dimension**

## Concatenating Along Dimension 0 (Rows)

Stack tensors **vertically** - adding more rows:

![Concat Dimension 0](/content/learn/tensors/concatenating-tensors/concat-dim0.png)

**Example:**

```python
import torch

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # Shape: (2, 3)

B = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])  # Shape: (2, 3)

# Concatenate along dimension 0 (rows)
result = torch.cat([A, B], dim=0)

print(result)
# tensor([[ 1,  2,  3],
#         [ 4,  5,  6],
#         [ 7,  8,  9],
#         [10, 11, 12]])

print(result.shape)  # torch.Size([4, 3])
```

**What happened:**

```yaml
A: (2, 3)  →  2 rows, 3 columns
B: (2, 3)  →  2 rows, 3 columns

Concatenate rows: 2 + 2 = 4 rows
Columns stay same: 3 columns

Result: (4, 3)
```

**Visual breakdown:**

```yaml
[[1, 2, 3],     ← From A
 [4, 5, 6],     ← From A
 [7, 8, 9],     ← From B
 [10, 11, 12]]  ← From B
```

## Concatenating Along Dimension 1 (Columns)

Join tensors **horizontally** - adding more columns:

![Concat Dimension 1](/content/learn/tensors/concatenating-tensors/concat-dim1.png)

**Example:**

```python
import torch

A = torch.tensor([[1, 2],
                  [3, 4]])  # Shape: (2, 2)

B = torch.tensor([[5, 6, 7],
                  [8, 9, 10]])  # Shape: (2, 3)

# Concatenate along dimension 1 (columns)
result = torch.cat([A, B], dim=1)

print(result)
# tensor([[ 1,  2,  5,  6,  7],
#         [ 3,  4,  8,  9, 10]])

print(result.shape)  # torch.Size([2, 5])
```

**What happened:**

```yaml
A: (2, 2)  →  2 rows, 2 columns
B: (2, 3)  →  2 rows, 3 columns

Rows stay same: 2 rows
Concatenate columns: 2 + 3 = 5 columns

Result: (2, 5)
```

**Visual breakdown:**

```yaml
[[1, 2,   5, 6, 7],
 [3, 4,   8, 9, 10]]
  ↑↑↑    ↑↑↑↑↑↑↑
  From A  From B
```

## The Concatenation Rules

![Concat Rules](/content/learn/tensors/concatenating-tensors/concat-rules.png)

**Rule:** All dimensions EXCEPT the concatenation dimension must match!

### ✓ Valid Examples

```python
# Concatenate dim=0: columns must match
A = torch.randn(2, 3)  # (2, 3)
B = torch.randn(4, 3)  # (4, 3) - same 3 columns ✓
result = torch.cat([A, B], dim=0)  # (6, 3)

# Concatenate dim=1: rows must match
C = torch.randn(5, 2)  # (5, 2)
D = torch.randn(5, 7)  # (5, 7) - same 5 rows ✓
result = torch.cat([C, D], dim=1)  # (5, 9)
```

### ✗ Invalid Examples

```python
# Different column counts - can't stack rows!
A = torch.randn(2, 3)
B = torch.randn(2, 4)  # Different columns
# torch.cat([A, B], dim=0)  # ERROR! 3 ≠ 4

# Different row counts - can't join columns!
C = torch.randn(3, 5)
D = torch.randn(2, 5)  # Different rows
# torch.cat([C, D], dim=1)  # ERROR! 3 ≠ 2
```

**Quick check:**

```yaml
Concatenating dim=0 (vertical):
  ✓ (2,3) + (4,3) → (6,3)  ← columns match (3)
  ✗ (2,3) + (2,4) → ERROR  ← columns don't match

Concatenating dim=1 (horizontal):
  ✓ (5,2) + (5,7) → (5,9)  ← rows match (5)
  ✗ (3,5) + (2,5) → ERROR  ← rows don't match
```

## Stack: Creating a New Dimension

`torch.stack()` is different - it **creates a new dimension**:

![Stack Visual](/content/learn/tensors/concatenating-tensors/stack-visual.png)

**Example:**

```python
import torch

A = torch.tensor([[1, 2], [3, 4]])  # (2, 2)
B = torch.tensor([[5, 6], [7, 8]])  # (2, 2)
C = torch.tensor([[9, 10], [11, 12]])  # (2, 2)

# Stack creates NEW dimension
stacked = torch.stack([A, B, C], dim=0)

print(stacked.shape)  # torch.Size([3, 2, 2])
# 3 matrices, each 2×2

print(stacked)
# tensor([[[ 1,  2],
#          [ 3,  4]],
#
#         [[ 5,  6],
#          [ 7,  8]],
#
#         [[ 9, 10],
#          [11, 12]]])
```

**Key difference:**

```yaml
cat([A, B], dim=0):
  (2, 3) + (2, 3) → (4, 3)  ← Adds to existing dimension
  
stack([A, B], dim=0):
  (2, 3) + (2, 3) → (2, 2, 3)  ← Creates NEW dimension
```

**For stack, all tensors must have EXACTLY the same shape!**

## Multiple Tensors at Once

You can concatenate more than 2 tensors:

```python
import torch

A = torch.ones(2, 3)
B = torch.ones(1, 3) * 2
C = torch.ones(3, 3) * 3

# Concatenate all three
result = torch.cat([A, B, C], dim=0)

print(result)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [2., 2., 2.],
#         [3., 3., 3.],
#         [3., 3., 3.],
#         [3., 3., 3.]])

print(result.shape)  # torch.Size([6, 3])
# 2 + 1 + 3 = 6 rows
```

**Breakdown:**

```yaml
A: 2 rows
B: 1 row
C: 3 rows

Total: 2 + 1 + 3 = 6 rows
```

## Practical Examples

### Example 1: Combining Train and Test Data

```python
import torch

# Training data: 100 samples
train_data = torch.randn(100, 10)

# Test data: 20 samples
test_data = torch.randn(20, 10)

# Combine into full dataset
full_data = torch.cat([train_data, test_data], dim=0)

print(full_data.shape)  # torch.Size([120, 10])
# 100 + 20 = 120 samples
```

### Example 2: Concatenating Features

```python
import torch

# Original features: 5 samples, 3 features each
original_features = torch.randn(5, 3)

# New features: 5 samples, 2 new features
new_features = torch.randn(5, 2)

# Combine features horizontally
combined = torch.cat([original_features, new_features], dim=1)

print(combined.shape)  # torch.Size([5, 5])
# 5 samples, 3 + 2 = 5 features
```

### Example 3: Creating Batches with Stack

```python
import torch

# Three separate samples
sample1 = torch.randn(28, 28)
sample2 = torch.randn(28, 28)
sample3 = torch.randn(28, 28)

# Stack into a batch
batch = torch.stack([sample1, sample2, sample3], dim=0)

print(batch.shape)  # torch.Size([3, 28, 28])
# 3 samples in the batch
```

### Example 4: Building Sequences

```python
import torch

# Word embeddings for a sentence
# Each word is a 100-dim vector
word1 = torch.randn(100)
word2 = torch.randn(100)
word3 = torch.randn(100)
word4 = torch.randn(100)

# Stack into sequence
sentence = torch.stack([word1, word2, word3, word4], dim=0)

print(sentence.shape)  # torch.Size([4, 100])
# 4 words, 100-dim embedding each
```

## Cat vs Stack

The key difference between `cat` and `stack`:

```python
import torch

A = torch.tensor([[1, 2], [3, 4]])  # (2, 2)
B = torch.tensor([[5, 6], [7, 8]])  # (2, 2)

# CAT: Joins along existing dimension
cat_result = torch.cat([A, B], dim=0)
print(cat_result.shape)  # torch.Size([4, 2])

# STACK: Creates new dimension
stack_result = torch.stack([A, B], dim=0)
print(stack_result.shape)  # torch.Size([2, 2, 2])
```

**When to use which:**

```yaml
Use cat() when:
  - Adding more samples to a batch
  - Extending features
  - Combining datasets
  - Tensors can have different sizes in concat dimension

Use stack() when:
  - Creating a batch from individual samples
  - All tensors have SAME shape
  - Want to add a new dimension
```

## Common Gotchas

### ❌ Gotcha 1: Shape Mismatch

```python
A = torch.randn(2, 3)
B = torch.randn(2, 4)

# This will ERROR!
# torch.cat([A, B], dim=0)  # 3 ≠ 4
```

### ❌ Gotcha 2: Wrong Dimension

```python
A = torch.randn(2, 3)
B = torch.randn(2, 3)

# This will ERROR!
# torch.cat([A, B], dim=2)  # Only dims 0 and 1 exist!
```

### ❌ Gotcha 3: Forgetting List Brackets

```python
A = torch.randn(2, 3)
B = torch.randn(2, 3)

# This will ERROR!
# torch.cat(A, B, dim=0)  # Missing [ ]

# Correct:
torch.cat([A, B], dim=0)  # ✓
```

## Key Takeaways

✓ **cat() joins along existing dimension:** Extends that dimension

✓ **stack() creates new dimension:** All tensors must have same shape

✓ **Other dimensions must match:** Can't concatenate incompatible shapes

✓ **dim=0 is vertical:** Stacks rows (more samples)

✓ **dim=1 is horizontal:** Joins columns (more features)

✓ **Use list brackets:** `torch.cat([A, B, C], dim=0)`

**Quick Reference:**

```python
# Concatenate (extends existing dimension)
torch.cat([A, B], dim=0)       # Stack vertically (more rows)
torch.cat([A, B], dim=1)       # Join horizontally (more columns)
torch.cat([A, B, C], dim=0)    # Multiple tensors

# Stack (creates new dimension)
torch.stack([A, B], dim=0)     # New dimension at front
torch.stack([A, B], dim=1)     # New dimension at position 1

# Split (opposite of concatenate)
torch.split(tensor, 2, dim=0)  # Split into chunks of size 2
torch.chunk(tensor, 3, dim=0)  # Split into 3 chunks
```

**Remember:** `cat()` extends, `stack()` creates! 🎉
