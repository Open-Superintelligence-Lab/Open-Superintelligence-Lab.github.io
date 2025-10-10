---
hero:
  title: "Creating Tensors"
  subtitle: "Building Blocks of Deep Learning"
  tags:
    - "🔢 Tensors"
    - "⏱️ 15 min read"
---

Tensors are the fundamental data structure in deep learning. Everything you work with in neural networks - images, text, audio, weights, gradients - is represented as tensors.

## What is a Tensor?

A **tensor** is a multi-dimensional array of numbers. Think of it as a container that can hold data in different dimensions:

- **0D Tensor (Scalar)**: A single number
- **1D Tensor (Vector)**: An array of numbers
- **2D Tensor (Matrix)**: A table of numbers
- **3D+ Tensor**: Multiple matrices stacked together

In PyTorch and other deep learning frameworks, tensors are similar to NumPy arrays but with superpowers - they can run on GPUs and automatically compute gradients!

![Tensor Dimensions](/content/learn/tensors/creating-tensors/tensor-dimensions.png)

## Understanding Tensor Dimensions

### 0D Tensor (Scalar)

A scalar is just a single number.

![Scalar Tensor](/content/learn/tensors/creating-tensors/scalar-tensor.png)

**Example:**

```python
import torch

# Creating a scalar tensor
scalar = torch.tensor(5)

print(scalar)           # Output: tensor(5)
print(scalar.shape)     # Output: torch.Size([])
print(scalar.ndim)      # Output: 0 (zero dimensions)
```

**Real-world use:** Learning rate, loss value, accuracy score

### 1D Tensor (Vector)

A vector is an array of numbers, like a list.

![Vector Tensor](/content/learn/tensors/creating-tensors/vector-tensor.png)

**Example 1:** Simple vector

```python
import torch

# Creating a 1D tensor (vector)
vector = torch.tensor([1, 2, 3, 4, 5])

print(vector)           # Output: tensor([1, 2, 3, 4, 5])
print(vector.shape)     # Output: torch.Size([5])
print(vector.ndim)      # Output: 1
```

**Example 2:** Accessing elements

```python
vector = torch.tensor([10, 20, 30, 40, 50])

# Access individual elements (0-indexed)
print(vector[0])        # Output: tensor(10)
print(vector[2])        # Output: tensor(30)
print(vector[-1])       # Output: tensor(50) (last element)

# Access a slice
print(vector[1:4])      # Output: tensor([20, 30, 40])
```

**Real-world use:** Word embeddings, feature vectors, time series data

### 2D Tensor (Matrix)

A matrix is a table of numbers with rows and columns.

![Matrix Tensor](/content/learn/tensors/creating-tensors/matrix-tensor.png)

**Example 1:** Creating a matrix

```python
import torch

# Creating a 2D tensor (matrix)
matrix = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

print(matrix)
# Output: 
# tensor([[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]])

print(matrix.shape)     # Output: torch.Size([3, 4])
                        # 3 rows, 4 columns
print(matrix.ndim)      # Output: 2
```

**Example 2:** Accessing rows and columns

```python
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# Access a single element [row, column]
print(matrix[0, 0])     # Output: tensor(1)
print(matrix[1, 2])     # Output: tensor(6)
print(matrix[2, 1])     # Output: tensor(8)

# Access entire row
print(matrix[0])        # Output: tensor([1, 2, 3])
print(matrix[1])        # Output: tensor([4, 5, 6])

# Access entire column
print(matrix[:, 0])     # Output: tensor([1, 4, 7])
print(matrix[:, 1])     # Output: tensor([2, 5, 8])
```

**Real-world use:** Grayscale images, batch of word embeddings, weight matrices

### 3D Tensor

A 3D tensor is multiple matrices stacked together. Think of it as a cube of numbers.

![3D Tensor](/content/learn/tensors/creating-tensors/3d-tensor.png)

**Example 1:** Creating a 3D tensor

```python
import torch

# Creating a 3D tensor (2 matrices, each 3x4)
tensor_3d = torch.tensor([[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]],
                          
                          [[13, 14, 15, 16],
                           [17, 18, 19, 20],
                           [21, 22, 23, 24]]])

print(tensor_3d.shape)  # Output: torch.Size([2, 3, 4])
                        # 2 matrices, each with 3 rows and 4 columns
print(tensor_3d.ndim)   # Output: 3
```

**Example 2:** Understanding shape (2, 3, 4)

- **First dimension (2)**: Number of matrices (or "depth")
- **Second dimension (3)**: Number of rows in each matrix
- **Third dimension (4)**: Number of columns in each matrix

```python
# Access the first matrix
print(tensor_3d[0])
# Output:
# tensor([[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]])

# Access the second matrix
print(tensor_3d[1])
# Output:
# tensor([[13, 14, 15, 16],
#         [17, 18, 19, 20],
#         [21, 22, 23, 24]])

# Access specific element [matrix, row, column]
print(tensor_3d[0, 1, 2])  # Output: tensor(7)
print(tensor_3d[1, 2, 3])  # Output: tensor(24)
```

**Real-world use:** RGB images (height, width, 3 color channels), video frames, batch of images

## Creating Tensors from Different Data Types

PyTorch provides multiple ways to create tensors from existing data.

![Creating from Data](/content/learn/tensors/creating-tensors/creating-from-data.png)

### From Python Lists

**Example 1:** 1D tensor from list

```python
import torch

# Create from Python list
python_list = [1, 2, 3, 4, 5]
tensor = torch.tensor(python_list)

print(tensor)           # Output: tensor([1, 2, 3, 4, 5])
print(type(tensor))     # Output: <class 'torch.Tensor'>
```

**Example 2:** 2D tensor from nested lists

```python
# Create 2D tensor from nested list
nested_list = [[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]]

tensor_2d = torch.tensor(nested_list)

print(tensor_2d)
# Output:
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])

print(tensor_2d.shape)  # Output: torch.Size([3, 3])
```

**Example 3:** 3D tensor from deeply nested lists

```python
# Create 3D tensor (2 matrices, each 2x3)
deep_list = [[[1, 2, 3],
              [4, 5, 6]],
             
             [[7, 8, 9],
              [10, 11, 12]]]

tensor_3d = torch.tensor(deep_list)

print(tensor_3d.shape)  # Output: torch.Size([2, 2, 3])
```

### From NumPy Arrays

If you're working with NumPy arrays, you can easily convert them to tensors.

**Example 1:** Converting NumPy array to tensor

```python
import torch
import numpy as np

# Create NumPy array
np_array = np.array([1, 2, 3, 4, 5])

# Convert to PyTorch tensor
tensor = torch.from_numpy(np_array)

print(np_array)         # Output: [1 2 3 4 5]
print(tensor)           # Output: tensor([1, 2, 3, 4, 5])
```

**Example 2:** 2D NumPy array to tensor

```python
# Create 2D NumPy array
np_matrix = np.array([[1, 2, 3],
                      [4, 5, 6]])

# Convert to tensor
tensor_from_np = torch.from_numpy(np_matrix)

print(tensor_from_np)
# Output:
# tensor([[1, 2, 3],
#         [4, 5, 6]])

print(tensor_from_np.shape)  # Output: torch.Size([2, 3])
```

**Important Note:** `torch.from_numpy()` shares memory with the original NumPy array, so changes to one affect the other!

```python
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)

# Modify NumPy array
np_array[0] = 999

print(np_array)         # Output: [999   2   3]
print(tensor)           # Output: tensor([999,   2,   3])
# They share memory!
```

### From Other Tensors

**Example:** Creating a new tensor with the same shape

```python
# Create original tensor
x = torch.tensor([[1, 2],
                  [3, 4]])

# Create new tensor with same shape (but different values)
y = torch.tensor([[5, 6],
                  [7, 8]])

print(x.shape)          # Output: torch.Size([2, 2])
print(y.shape)          # Output: torch.Size([2, 2])
```

## Specifying Data Types

Tensors can hold different types of numbers. Choosing the right data type is important for memory efficiency and computation speed.

![Data Types](/content/learn/tensors/creating-tensors/data-types.png)

### Common Data Types

- `torch.int32` or `torch.int`: 32-bit integers (4 bytes per number)
- `torch.int64` or `torch.long`: 64-bit integers (8 bytes per number)
- `torch.float32` or `torch.float`: 32-bit floating point (4 bytes per number) **[Most Common]**
- `torch.float64` or `torch.double`: 64-bit floating point (8 bytes per number)
- `torch.bool`: Boolean values (True/False)

**Example 1:** Creating tensors with specific data types

```python
import torch

# Integer tensor (int32)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
print(int_tensor)       # Output: tensor([1, 2, 3], dtype=torch.int32)
print(int_tensor.dtype) # Output: torch.int32

# Float tensor (float32) - Most common for neural networks!
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print(float_tensor)     # Output: tensor([1., 2., 3.])
print(float_tensor.dtype)  # Output: torch.float32

# Boolean tensor
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)
print(bool_tensor)      # Output: tensor([ True, False,  True])
print(bool_tensor.dtype)   # Output: torch.bool
```

**Example 2:** Default data type behavior

```python
# PyTorch infers the data type from your input

# Integers → int64 (by default)
x = torch.tensor([1, 2, 3])
print(x.dtype)          # Output: torch.int64

# Floats → float32 (by default)
y = torch.tensor([1.0, 2.0, 3.0])
print(y.dtype)          # Output: torch.float32

# Mixed integers and floats → float32
z = torch.tensor([1, 2.0, 3])
print(z)                # Output: tensor([1., 2., 3.])
print(z.dtype)          # Output: torch.float32
```

**Example 3:** Converting between data types

```python
# Create integer tensor
int_tensor = torch.tensor([1, 2, 3])
print(int_tensor.dtype)  # Output: torch.int64

# Convert to float
float_tensor = int_tensor.float()
print(float_tensor)      # Output: tensor([1., 2., 3.])
print(float_tensor.dtype)  # Output: torch.float32

# Convert back to int
back_to_int = float_tensor.int()
print(back_to_int.dtype)   # Output: torch.int32

# Alternative syntax
also_float = int_tensor.to(torch.float32)
print(also_float.dtype)    # Output: torch.float32
```

**Example 4:** Why data type matters

```python
# Memory usage comparison
large_int64 = torch.ones(1000000, dtype=torch.int64)
large_int32 = torch.ones(1000000, dtype=torch.int32)

print(f"int64 tensor: {large_int64.element_size() * large_int64.nelement() / 1e6} MB")
# Output: 8.0 MB (8 bytes per element)

print(f"int32 tensor: {large_int32.element_size() * large_int32.nelement() / 1e6} MB")
# Output: 4.0 MB (4 bytes per element)

# int32 uses half the memory!
```

## Step-by-Step Tensor Creation Examples

Let's walk through complete examples from start to finish.

### Example 1: Creating a Temperature Dataset

Imagine you recorded temperatures for 7 days.

```python
import torch

# Step 1: Your raw data (in Celsius)
temperatures = [20, 22, 19, 21, 23, 25, 24]

# Step 2: Convert to tensor
temp_tensor = torch.tensor(temperatures, dtype=torch.float32)

print("Temperatures:", temp_tensor)
# Output: Temperatures: tensor([20., 22., 19., 21., 23., 25., 24.])

print("Shape:", temp_tensor.shape)
# Output: Shape: torch.Size([7])

print("Data type:", temp_tensor.dtype)
# Output: Data type: torch.float32

# Step 3: Calculate average temperature
average_temp = temp_tensor.mean()
print(f"Average temperature: {average_temp:.2f}°C")
# Output: Average temperature: 22.00°C
```

### Example 2: Creating a Grayscale Image

A small 3x3 grayscale image where each pixel has a value from 0 (black) to 255 (white).

```python
import torch

# Step 1: Define pixel values (3x3 image)
# Values: 0 = black, 255 = white
image_data = [[0, 128, 255],      # Top row: black, gray, white
              [64, 128, 192],      # Middle row
              [255, 128, 0]]       # Bottom row: white, gray, black

# Step 2: Create 2D tensor
image_tensor = torch.tensor(image_data, dtype=torch.float32)

print("Image tensor:")
print(image_tensor)
# Output:
# tensor([[  0., 128., 255.],
#         [ 64., 128., 192.],
#         [255., 128.,   0.]])

print("\nShape:", image_tensor.shape)
# Output: Shape: torch.Size([3, 3])

# Step 3: Access specific pixels
top_left_pixel = image_tensor[0, 0]
center_pixel = image_tensor[1, 1]

print(f"\nTop-left pixel value: {top_left_pixel}")
# Output: Top-left pixel value: 0.0 (black)

print(f"Center pixel value: {center_pixel}")
# Output: Center pixel value: 128.0 (gray)
```

### Example 3: Creating a Batch of Vectors

In deep learning, we often process multiple examples at once (a "batch").

```python
import torch

# Step 1: Create 4 examples, each with 3 features
# Example: [height, weight, age]
person_1 = [170, 65, 25]  # 170cm, 65kg, 25 years
person_2 = [180, 80, 30]
person_3 = [165, 55, 22]
person_4 = [175, 70, 28]

# Step 2: Stack them into a batch (2D tensor)
batch = torch.tensor([person_1, person_2, person_3, person_4], 
                     dtype=torch.float32)

print("Batch of data:")
print(batch)
# Output:
# tensor([[170.,  65.,  25.],
#         [180.,  80.,  30.],
#         [165.,  55.,  22.],
#         [175.,  70.,  28.]])

print("\nShape:", batch.shape)
# Output: Shape: torch.Size([4, 3])
# 4 people, 3 features each

# Step 3: Access data
all_heights = batch[:, 0]    # All heights (first column)
all_weights = batch[:, 1]    # All weights (second column)
all_ages = batch[:, 2]       # All ages (third column)

print(f"\nAll heights: {all_heights}")
# Output: All heights: tensor([170., 180., 165., 175.])

print(f"Average height: {all_heights.mean():.2f}cm")
# Output: Average height: 172.50cm

print(f"Average weight: {all_weights.mean():.2f}kg")
# Output: Average weight: 67.50kg
```

### Example 4: Creating RGB Image Data

A tiny 2x2 RGB color image (3 color channels).

```python
import torch

# Step 1: Define a 2x2 RGB image
# Shape will be: (height=2, width=2, channels=3)
# Each pixel has [Red, Green, Blue] values from 0-255

image_rgb = [
    # Row 1
    [[255, 0, 0],    [0, 255, 0]],    # Red pixel, Green pixel
    # Row 2  
    [[0, 0, 255],    [255, 255, 0]]   # Blue pixel, Yellow pixel
]

# Step 2: Create 3D tensor
rgb_tensor = torch.tensor(image_rgb, dtype=torch.float32)

print("RGB Image tensor:")
print(rgb_tensor)

print("\nShape:", rgb_tensor.shape)
# Output: Shape: torch.Size([2, 2, 3])
# 2 height, 2 width, 3 color channels

# Step 3: Access specific pixels and channels
top_left_pixel = rgb_tensor[0, 0]
print(f"\nTop-left pixel (Red): {top_left_pixel}")
# Output: tensor([255.,   0.,   0.])

# Access just the red channel of all pixels
red_channel = rgb_tensor[:, :, 0]
print(f"\nRed channel:\n{red_channel}")
# Output:
# tensor([[255.,   0.],
#         [  0., 255.]])
```

## Common Mistakes and How to Fix Them

### Mistake 1: Shape Mismatch

```python
# ❌ Wrong: Inconsistent row lengths
try:
    wrong_tensor = torch.tensor([[1, 2, 3],
                                 [4, 5]])  # Second row too short!
except:
    print("Error: All rows must have the same length")

# ✅ Correct: All rows same length
correct_tensor = torch.tensor([[1, 2, 3],
                               [4, 5, 6]])
print(correct_tensor.shape)  # Output: torch.Size([2, 3])
```

### Mistake 2: Mixing Data Types Unintentionally

```python
# ❌ Be careful with integer division
int_tensor = torch.tensor([10, 20, 30])
result = int_tensor / 3
print(result)  # Output: tensor([3.3333, 6.6667, 10.0000])
print(result.dtype)  # Output: torch.float32 (changed to float!)

# ✅ If you want to keep integers, use integer division
int_result = int_tensor // 3
print(int_result)  # Output: tensor([3, 6, 10])
print(int_result.dtype)  # Output: torch.int64
```

### Mistake 3: Forgetting Dimension Order

```python
# For images, be careful about dimension order!

# ❌ Wrong order: (channels, height, width)
# This might cause errors in some operations
wrong_order = torch.rand(3, 224, 224)  

# ✅ PyTorch usually expects: (batch, channels, height, width)
correct_batch = torch.rand(1, 3, 224, 224)  # 1 image, 3 channels, 224x224

# ✅ For a single image: (channels, height, width)
single_image = torch.rand(3, 224, 224)
```

## Quick Reference

### Creating Tensors

```python
# From list
torch.tensor([1, 2, 3])

# From NumPy
torch.from_numpy(np_array)

# With specific dtype
torch.tensor([1, 2], dtype=torch.float32)
```

### Checking Tensor Properties

```python
tensor = torch.tensor([[1, 2], [3, 4]])

tensor.shape      # Shape: torch.Size([2, 2])
tensor.size()     # Same as .shape
tensor.ndim       # Number of dimensions: 2
tensor.dtype      # Data type: torch.int64
tensor.numel()    # Total number of elements: 4
```

### Data Type Conversion

```python
tensor.float()    # Convert to float32
tensor.int()      # Convert to int32
tensor.long()     # Convert to int64
tensor.double()   # Convert to float64
tensor.bool()     # Convert to boolean
```

## Why Tensors Matter for Neural Networks

- **Images**: RGB images are 3D tensors (height × width × 3 channels)
- **Batches**: Neural networks process multiple examples at once (batch dimension)
- **Text**: Word embeddings are 2D tensors (sequence length × embedding dimension)
- **Weights**: Model parameters are tensors that get updated during training

**Example: A batch of images**
```python
# Shape: (batch_size, channels, height, width)
batch_of_images = torch.rand(32, 3, 224, 224)
# 32 images, 3 color channels (RGB), 224×224 pixels

print(f"Batch shape: {batch_of_images.shape}")
# Output: Batch shape: torch.Size([32, 3, 224, 224])

print(f"Total numbers in this batch: {batch_of_images.numel():,}")
# Output: Total numbers in this batch: 4,816,896
```

**Congratulations! You now understand how to create and work with tensors!** 🎉

In the next lessons, we'll learn how to create special tensors (zeros, ones, random values) and perform operations on them.
