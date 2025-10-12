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

Let's look at two successful concatenation operations. The first one stacks tensors vertically by adding more rows, while the second joins them horizontally by adding more columns:

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

Notice how in the first case, A and B both have 3 columns (the dimension we're NOT concatenating along), and in the second case, C and D both have 5 rows. This matching is what makes the concatenation valid!

### ✗ Invalid Examples

Now let's see what happens when the non-concatenating dimensions don't match. These will produce errors:

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

You're not limited to joining just two tensors - you can concatenate as many as you need in a single operation. This is really useful when you're combining data from multiple sources.

Let's concatenate three tensors, each with different values so we can see where each piece ends up:

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

The tensors are joined in the order they appear in the list: first A's rows, then B's row, then C's rows.

**Breakdown:**

```yaml
A: 2 rows
B: 1 row
C: 3 rows

Total: 2 + 1 + 3 = 6 rows
```

## Practical Examples

Let's see how concatenation is used in real machine learning scenarios!

### Example 1: Combining Train and Test Data

When you want to analyze your entire dataset together, or apply the same preprocessing to both training and test data, you can concatenate them:

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

We concatenate along `dim=0` because we're adding more samples (rows), not more features. Both datasets have 10 features, which stay the same.

### Example 2: Concatenating Features

Sometimes you want to augment your data with additional features. This means adding more columns to your existing data:

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

Here we use `dim=1` because we're extending the features (columns), not adding more samples. The number of rows (5 samples) remains constant.

### Example 3: Creating Batches with Stack

When training neural networks, you often need to group individual samples into batches. `stack()` is perfect for this because it creates a new batch dimension:

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

The result is a 3D tensor where the first dimension is the batch size (3), and each "slice" is one 28×28 image.

### Example 4: Building Sequences

In natural language processing, you need to stack word embeddings into sequences. Each word is represented as a vector, and a sentence is a sequence of these vectors:

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

This creates a 2D tensor where each row is one word's embedding vector. The shape tells us we have a 4-word sentence with 100-dimensional embeddings.

## Cat vs Stack

Let's directly compare these two operations to understand when to use each one. The key difference between `cat` and `stack`:

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

See the difference? `cat()` made the tensor taller (4 rows instead of 2), while `stack()` added a whole new dimension, creating a 3D tensor!

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

Here are the most common mistakes people make when concatenating tensors. Knowing these will save you debugging time!

### ❌ Gotcha 1: Shape Mismatch

The most common error is trying to concatenate tensors whose non-concatenating dimensions don't match:

```python
A = torch.randn(2, 3)
B = torch.randn(2, 4)

# This will ERROR!
# torch.cat([A, B], dim=0)  # 3 ≠ 4
```

Remember: when concatenating along `dim=0`, the number of columns must match. Here A has 3 columns but B has 4 - PyTorch can't line them up!

### ❌ Gotcha 2: Wrong Dimension

Trying to concatenate along a dimension that doesn't exist:

```python
A = torch.randn(2, 3)
B = torch.randn(2, 3)

# This will ERROR!
# torch.cat([A, B], dim=2)  # Only dims 0 and 1 exist!
```

A 2D tensor only has dimensions 0 and 1 (rows and columns). Asking for `dim=2` is like asking for a direction that doesn't exist!

### ❌ Gotcha 3: Forgetting List Brackets

A syntax error that catches many beginners:

```python
A = torch.randn(2, 3)
B = torch.randn(2, 3)

# This will ERROR!
# torch.cat(A, B, dim=0)  # Missing [ ]

# Correct:
torch.cat([A, B], dim=0)  # ✓
```

The tensors must be in a list (inside square brackets). This is because `torch.cat()` is designed to handle any number of tensors, so it expects them as a single list argument.

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
