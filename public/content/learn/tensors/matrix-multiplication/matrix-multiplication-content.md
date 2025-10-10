---
hero:
  title: "Matrix Multiplication"
  subtitle: "The Core Operation in Neural Networks"
  tags:
    - "🔢 Tensors"
    - "⏱️ 10 min read"
---

Matrix multiplication is THE most important operation in deep learning. Unlike addition, it's **not element-wise** - it combines rows and columns in a special way.

## The Key Difference

**Addition:** Add each position separately  
**Multiplication:** Combine entire rows with entire columns

Let's build up to matrix multiplication step by step!

## Step 1: The Dot Product

Before matrices, let's understand the **dot product** - multiplying two vectors:

![Dot Product](/content/learn/tensors/matrix-multiplication/dot-product.png)

**Example:**

```python
import torch

a = torch.tensor([2, 3, 4])
b = torch.tensor([1, 2, 3])

# Dot product
result = torch.dot(a, b)

print(result)  # Output: tensor(20)
```

**Manual calculation:**

```yaml
Step 1: Multiply corresponding elements
2 × 1 = 2
3 × 2 = 6
4 × 3 = 12

Step 2: Add them all up
2 + 6 + 12 = 20

Result: 20
```

**Key insight:** Dot product = multiply pairs, then sum everything.

![Dot Product Steps](/content/learn/tensors/matrix-multiplication/dot-product-steps.png)

## Step 2: Matrix @ Matrix

Matrix multiplication uses dot products repeatedly! The `@` operator means "matrix multiply":

![Simple Matrix Multiplication](/content/learn/tensors/matrix-multiplication/simple-matmul.png)

**Example:**

```python
import torch

A = torch.tensor([[1, 2],
                  [3, 4]])

B = torch.tensor([[5, 6],
                  [7, 8]])

result = A @ B  # @ means matrix multiply

print(result)
# Output:
# tensor([[19, 22],
#         [43, 50]])
```

**How does this work?** Each position in the result is a dot product!

## Computing One Position: The Rule

**To get result[row, col]:**
1. Take the **row** from matrix A
2. Take the **column** from matrix B
3. Compute their **dot product**

![Step by Step](/content/learn/tensors/matrix-multiplication/step-by-step.png)

**Manual calculation for position [0, 0]:**

```yaml
Take row 0 from A:  [1, 2]
Take column 0 from B:  [5, 7]

Dot product:
(1 × 5) + (2 × 7) = 5 + 14 = 19

Result[0, 0] = 19
```

**Manual calculation for position [0, 1]:**

```yaml
Take row 0 from A:  [1, 2]
Take column 1 from B:  [6, 8]

Dot product:
(1 × 6) + (2 × 8) = 6 + 16 = 22

Result[0, 1] = 22
```

**Manual calculation for position [1, 0]:**

```yaml
Take row 1 from A:  [3, 4]
Take column 0 from B:  [5, 7]

Dot product:
(3 × 5) + (4 × 7) = 15 + 28 = 43

Result[1, 0] = 43
```

**Manual calculation for position [1, 1]:**

```yaml
Take row 1 from A:  [3, 4]
Take column 1 from B:  [6, 8]

Dot product:
(3 × 6) + (4 × 8) = 18 + 32 = 50

Result[1, 1] = 50
```

**Complete result:**

```yaml
[[19, 22],
 [43, 50]]
```

![All Positions](/content/learn/tensors/matrix-multiplication/all-positions.png)

## The Shape Rule

**Not all matrices can be multiplied!** The shapes must be compatible:

![Shape Rule](/content/learn/tensors/matrix-multiplication/shape-rule.png)

**The rule:** `(m, n) @ (n, p) = (m, p)`

The **inner dimensions must match**!

### ✓ Valid Examples

```python
# Example 1
A = torch.randn(3, 4)  # 3 rows, 4 columns
B = torch.randn(4, 2)  # 4 rows, 2 columns
result = A @ B         # Works! → (3, 2)

# Example 2
A = torch.randn(5, 10)
B = torch.randn(10, 7)
result = A @ B         # Works! → (5, 7)

# Example 3
A = torch.randn(2, 3)
B = torch.randn(3, 3)
result = A @ B         # Works! → (2, 3)
```

**Why these work:**

```yaml
Example 1: (3, 4) @ (4, 2) = (3, 2)  ✓ 4 = 4
Example 2: (5, 10) @ (10, 7) = (5, 7)  ✓ 10 = 10
Example 3: (2, 3) @ (3, 3) = (2, 3)  ✓ 3 = 3
```

### ✗ Invalid Examples

```python
# Example 1 - WILL ERROR!
A = torch.randn(3, 4)
B = torch.randn(5, 2)
# result = A @ B  # Error! 4 ≠ 5

# Example 2 - WILL ERROR!
A = torch.randn(2, 7)
B = torch.randn(3, 5)
# result = A @ B  # Error! 7 ≠ 3
```

**Why these fail:**

```yaml
Example 1: (3, 4) @ (5, 2)  ✗ 4 ≠ 5 (can't match rows with columns)
Example 2: (2, 7) @ (3, 5)  ✗ 7 ≠ 3 (dimensions incompatible)
```

## Vector @ Matrix

A common pattern in neural networks is multiplying a vector by a matrix:

![Vector @ Matrix](/content/learn/tensors/matrix-multiplication/vector-matrix.png)

**Example:**

```python
import torch

# Input vector (like data going into a layer)
x = torch.tensor([1, 2, 3])  # Shape: (3,)

# Weight matrix
W = torch.tensor([[4, 5],
                  [6, 7],
                  [8, 9]])   # Shape: (3, 2)

result = x @ W

print(result)  # Output: tensor([40, 46])
print(result.shape)  # Shape: (2,)
```

**Manual calculation:**

```yaml
Position [0]:
Take vector:  [1, 2, 3]
Take column 0:  [4, 6, 8]
Dot product: (1×4) + (2×6) + (3×8) = 4 + 12 + 24 = 40

Position [1]:
Take vector:  [1, 2, 3]
Take column 1:  [5, 7, 9]
Dot product: (1×5) + (2×7) + (3×9) = 5 + 14 + 27 = 46

Result: [40, 46]
```

**This is exactly what happens in a neural network layer!**

## Practical Example: Neural Network Layer

Here's a realistic example of matrix multiplication in action:

```python
import torch

# Batch of 2 samples, each with 3 features
inputs = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])  # Shape: (2, 3)

# Weight matrix: 3 inputs → 4 outputs
weights = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7, 0.8],
                        [0.9, 1.0, 1.1, 1.2]])  # Shape: (3, 4)

# Forward pass
outputs = inputs @ weights  # Shape: (2, 4)

print(outputs)
# tensor([[3.2000, 3.8000, 4.4000, 5.0000],
#         [7.7000, 9.2000, 10.7000, 12.2000]])
```

**What happened:**

```yaml
Shape: (2, 3) @ (3, 4) = (2, 4)
       ↓       ↓       ↓
    2 samples  →  4 outputs per sample
    3 features each
```

Each of the 2 input samples got transformed into 4 output values. This is how neural networks transform data!

![Neural Network Layer](/content/learn/tensors/matrix-multiplication/neural-network.png)

## Matrix @ Vector

You can also multiply matrix @ vector (different from vector @ matrix):

```python
import torch

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # Shape: (2, 3)

v = torch.tensor([1, 2, 3])   # Shape: (3,)

result = A @ v

print(result)  # Output: tensor([14, 32])
print(result.shape)  # Shape: (2,)
```

**Manual calculation:**

```yaml
Row 0: [1, 2, 3] · [1, 2, 3] = 1 + 4 + 9 = 14
Row 1: [4, 5, 6] · [1, 2, 3] = 4 + 10 + 18 = 32

Result: [14, 32]
```

## Common Mistakes

### ❌ Mistake 1: Using * instead of @

```python
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

wrong = A * B    # Element-wise multiplication! ❌
right = A @ B    # Matrix multiplication! ✓

print("Wrong:", wrong)
# tensor([[ 5, 12],
#         [21, 32]])

print("Right:", right)
# tensor([[19, 22],
#         [43, 50]])
```

**Visual comparison:**

![Element-wise vs Matrix Multiplication](/content/learn/tensors/matrix-multiplication/elementwise-vs-matmul.png)

### ❌ Mistake 2: Wrong shape order

```python
A = torch.randn(3, 4)
B = torch.randn(5, 3)

# result = A @ B  # Error! 4 ≠ 5

# Fix: Either change order or transpose
result = B @ A  # Works! (5, 3) @ (3, 4) = (5, 4)
```

## Key Takeaways

✓ **Dot product:** Multiply pairs, then sum

✓ **Matrix multiply:** Each result position = dot product of row × column

✓ **Shape rule:** `(m, n) @ (n, p) = (m, p)` - inner dimensions must match!

✓ **Use @:** For matrix multiplication (not `*`)

✓ **Common in ML:** Input @ Weights = Output

**Quick Reference:**

```python
# Dot product (1D × 1D)
torch.dot(torch.tensor([1, 2]), torch.tensor([3, 4]))  # = 11

# Vector @ Matrix (transforms vector)
torch.tensor([1, 2]) @ torch.tensor([[1, 2], [3, 4]])  # = [7, 10]

# Matrix @ Vector (applies to rows)
torch.tensor([[1, 2], [3, 4]]) @ torch.tensor([1, 2])  # = [5, 11]

# Matrix @ Matrix (transforms matrix)
torch.tensor([[1, 2], [3, 4]]) @ torch.tensor([[5, 6], [7, 8]])
# = [[19, 22], [43, 50]]
```

**Remember:** Every neural network layer uses matrix multiplication to transform data. You've just learned the most important operation in deep learning! 🎉
