---
hero:
  title: "Tensor Addition"
  subtitle: "Element-wise Operations on Tensors"
  tags:
    - "🔢 Tensors"
    - "⏱️ 8 min read"
---

Tensor addition is one of the most fundamental operations in deep learning. It's simple: **add corresponding elements together**.

## The Basic Rule

**When you add two tensors, you add each position separately (element-wise).**

Think of it like adding two shopping lists item by item:
- First item + First item
- Second item + Second item  
- Third item + Third item

## Scalar Addition

Adding two single numbers:

![Scalar Addition](/content/learn/tensors/tensor-addition/scalar-addition.png)

**Example:**

```python
import torch

a = torch.tensor(5)
b = torch.tensor(3)
result = a + b

print(result)  # Output: tensor(8)
```

**Manual calculation:**
```yaml
5 + 3 = 8
```

Simple! Just like regular math.

## Vector Addition

Adding arrays of numbers, **element by element**:

![Vector Addition](/content/learn/tensors/tensor-addition/vector-addition.png)

**Example:**

```python
import torch

a = torch.tensor([10, 20, 30])
b = torch.tensor([5, 15, 25])
result = a + b

print(result)  # Output: tensor([15, 35, 55])
```

**Manual calculation:**
```yaml
Position 0: 10 + 5  = 15
Position 1: 20 + 15 = 35
Position 2: 30 + 25 = 55

Result: [15, 35, 55]
```

![Step by Step Addition](/content/learn/tensors/tensor-addition/step-by-step-addition.png)

**Key insight:** Each position is independent. We add `[0]` with `[0]`, `[1]` with `[1]`, `[2]` with `[2]`.

## Matrix Addition

Same rule applies to matrices - add corresponding positions:

![Matrix Addition](/content/learn/tensors/tensor-addition/matrix-addition.png)

**Example:**

```python
import torch

a = torch.tensor([[10, 20, 30],
                  [15, 25, 35]])

b = torch.tensor([[5, 10, 15],
                  [8, 12, 18]])

result = a + b

print(result)
# Output:
# tensor([[15, 30, 45],
#         [23, 37, 53]])
```

**Manual calculation:**
```yaml
Position [0, 0]: 10 + 5  = 15
Position [0, 1]: 20 + 10 = 30
Position [0, 2]: 30 + 15 = 45
Position [1, 0]: 15 + 8  = 23
Position [1, 1]: 25 + 12 = 37
Position [1, 2]: 35 + 18 = 53

Result:
[[15, 30, 45],
 [23, 37, 53]]
```

## Broadcasting: Adding a Scalar to a Vector

What if you want to add a single number to every element in a vector? PyTorch automatically "broadcasts" the scalar:

![Broadcasting](/content/learn/tensors/tensor-addition/broadcasting-scalar-vector.png)

**Example:**

```python
import torch

vector = torch.tensor([10, 20, 30])
scalar = 5

result = vector + scalar

print(result)  # Output: tensor([15, 25, 35])
```

**What happens behind the scenes:**

PyTorch automatically expands `5` to `[5, 5, 5]` and then adds:

```yaml
[10, 20, 30] + 5
    ↓
[10, 20, 30] + [5, 5, 5]  (automatic broadcast)
    ↓
[15, 25, 35]
```

**Manual calculation:**
```yaml
10 + 5 = 15
20 + 5 = 25
30 + 5 = 35

Result: [15, 25, 35]
```

This works because adding the same number to every position makes sense!

## Addition Rules

### Quick Reminder: What is Shape?

- **Shape** tells you the dimensions and size of your tensor
- Written as `(rows, columns)` for 2D, or `(size,)` for 1D

**Examples:**
```yaml
5             → Shape: ()         (scalar - no dimensions)
[1, 2, 3]     → Shape: (3,)       (1D - 3 elements)
[[1, 2],      → Shape: (3, 2)     (2D - 3 rows, 2 columns) - last shape number is the inner most tensor dimension
 [3, 4],
 [5, 6]]
[[[...],      → Shape: (2, 3, 5)  (3D - 2 matrices, 3 rows, 5 columns)
  [...],
  [...]],
 [[...],
  [...],
  [...]]]

...and so on for higher dimensions
```

Now let's use this to understand addition rules!

### ✓ Rule 1: Same Shapes Work

Tensors must have the **same shape** to be added:

```python
a = torch.tensor([1, 2, 3])      # Shape: (3,)
b = torch.tensor([4, 5, 6])      # Shape: (3,)
result = a + b                    # Works! ✓

print(result)  # tensor([5, 7, 9])
```

### ✓ Rule 2: Broadcasting Works

A scalar can be added to any tensor:

```python
a = torch.tensor([1, 2, 3])      # Shape: (3,)
b = 10                            # Scalar
result = a + b                    # Works! ✓

print(result)  # tensor([11, 12, 13])
```

### ✗ Rule 3: Different Shapes Don't Work

You **cannot** add tensors with different shapes:

```python
a = torch.tensor([1, 2, 3])      # Shape: (3,)
b = torch.tensor([4, 5])         # Shape: (2,)

# This will cause an ERROR! ✗
# result = a + b  
# RuntimeError: The size of tensor a (3) must match the size of tensor b (2)
```

**Why?** PyTorch doesn't know how to match up the elements:
- Should position `[0]` add to `[0]`? Yes.
- Should position `[1]` add to `[1]`? Yes.  
- Should position `[2]` add to...? There's no `[2]` in the second tensor! ❌

## Real-World Example: Adjusting Image Brightness

Imagine you have a small 2×2 grayscale image (values 0-255):

```python
import torch

# Original image (darker)
image = torch.tensor([[100, 150],
                      [120, 180]], dtype=torch.float32)

# Make it brighter by adding 50 to all pixels
brightness_increase = 50
brighter_image = image + brightness_increase

print("Original image:")
print(image)
# tensor([[100., 150.],
#         [120., 180.]])

print("\nBrighter image:")
print(brighter_image)
# tensor([[150., 200.],
#         [170., 230.]])
```

**Manual calculation:**
```yaml
Original:     Add 50:      Result:
[[100, 150]   +  50    →   [[150, 200]
 [120, 180]]                 [170, 230]]

Each pixel becomes 50 points brighter!
```

This is exactly how image editing software makes images brighter - it adds a value to every pixel.

## Key Takeaways

✓ **Element-wise:** Addition happens position by position

✓ **Same shapes:** Tensors must have identical shapes (or use broadcasting)

✓ **Broadcasting:** Scalars are automatically added to every element

✓ **Independent:** Each position is added separately - no mixing between positions

**Quick Reference:**

```python
# Scalar + Scalar
torch.tensor(5) + torch.tensor(3)          # = 8

# Vector + Vector (same size)
torch.tensor([1, 2]) + torch.tensor([3, 4])  # = [4, 6]

# Vector + Scalar (broadcasting)
torch.tensor([1, 2, 3]) + 10               # = [11, 12, 13]

# Matrix + Matrix (same shape)
torch.tensor([[1, 2], [3, 4]]) + torch.tensor([[5, 6], [7, 8]])
# = [[6, 8], [10, 12]]
```

**Congratulations!** You now understand tensor addition. This same element-wise principle applies to subtraction, multiplication, and division too! 🎉
