---
hero:
  title: "Functions"
  subtitle: "Building Blocks of Neural Networks"
  tags:
    - "📐 Mathematics"
    - "⏱️ 12 min read"
---

Functions are the foundation of neural networks.

## What is a Function?

In simple terms, function is like a machine that takes something in and gives something back out. More formally, a **function** is a mathematical relationship that **maps inputs to outputs**.



## Simple Examples

### Example 1: Linear Function f(x) = 2x + 3

This is a function that takes any number x and returns 2x + 3.

![Linear Function](/content/learn/math/functions/linear-function.png)

Let's calculate f(x) for different values step by step:

For x = 1:





f(1) = 2(1) + 3 = 2 + 3 = 5



Don't confuse `f(1)` and `2(1)`. `f(1)` means passing 1 into function f, and `2(1)` mean `2*1`.

For x = 0:





f(0) = 2(0) + 3 = 0 + 3 = 3

For x = -1:





f(-1) = 2(-1) + 3 = -2 + 3 = 1

Now image a function that takes in "Cat sat on a" and returns "mat" - that function would be a lot more difficult to create, but neural networks (LLMs) can learn it.

### Example 2: Quadratic Function f(x) = x² + 2x + 1

![Quadratic Function](/content/learn/math/functions/quadratic-function.png)

Let's calculate f(x) for different values step by step:

For x = 2:





f(2) = (2)² + 2(2) + 1 = 4 + 4 + 1 = 9

For x = 0:





f(0) = (0)² + 2(0) + 1 = 0 + 0 + 1 = 1

For x = -1:





f(-1) = (-1)² + 2(-1) + 1 = 1 - 2 + 1 = 0

## Mathematical Definition of a Function

A function **f: A → B** maps every element in set A to **exactly one** element in set B.

Previous quadratic function will always give 9 if x=2 and nothing else.

## Notation





**f(x) = y** (read as "f of x equals y")



**x** is the input (independent variable)



**y** is the output (dependent variable) - it depends on x

## Code Examples

Our 2 functions coded in python, if you are unfamiliar with python you can skip the code, next module will focus on python.

```python
# Linear function: f(x) = 2x + 3
def linear_function(x):
    return 2 * x + 3

# Test the function
print(f"f(1) = {linear_function(1)}")  # Output: f(1) = 5
print(f"f(0) = {linear_function(0)}")  # Output: f(0) = 3
print(f"f(-1) = {linear_function(-1)}")  # Output: f(-1) = 1

# Quadratic function: f(x) = x² + 2x + 1
def quadratic_function(x):
    return x**2 + 2*x + 1

# Test the function
print(f"f(2) = {quadratic_function(2)}")  # Output: f(2) = 9
print(f"f(0) = {quadratic_function(0)}")  # Output: f(0) = 1
print(f"f(-1) = {quadratic_function(-1)}")  # Output: f(-1) = 0
```

## Types of Functions

### 1. Linear Functions

Linear functions have the form: **f(x) = mx + b**

Where:





**m** is the slope (how steep the line is)



**b** is the y-intercept (where the line crosses the y-axis)

Let's draw it

![Linear Functions Comparison](/content/learn/math/functions/linear-functions-comparison.png)

Blue line: 2x + 1





2 is the slope, meaning that if you move by 1 on x axis, y will go up by 2



y or f(x) - it's the same



1 is the value on y coordinate where the blue line will cross it (y-intercept), at x=0 - see it for yourself, blue line should pass through x=0 and y=1 spot

### 2. Polynomial Functions

Functions with powers of x: **f(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀**

**Hand Calculation Examples**

**Example: f(x) = x³ - 3x² + 2x + 1**

Let's calculate f(x) for different values step by step:

For x = 1:





f(1) = (1)³ - 3(1)² + 2(1) + 1



f(1) = 1 - 3(1) + 2 + 1



f(1) = 1 - 3 + 2 + 1



f(1) = 1

For x = 2:





f(2) = (2)³ - 3(2)² + 2(2) + 1



f(2) = 8 - 3(4) + 4 + 1



f(2) = 8 - 12 + 4 + 1



f(2) = 1

For x = 0:





f(0) = (0)³ - 3(0)² + 2(0) + 1



f(0) = 0 - 0 + 0 + 1



f(0) = 1

**Example: f(x) = x⁴ - 4x² + 3**

Let's calculate f(x) for different values step by step:

For x = 1:





f(1) = (1)⁴ - 4(1)² + 3



f(1) = 1 - 4(1) + 3



f(1) = 1 - 4 + 3



f(1) = 0

For x = 2:





f(2) = (2)⁴ - 4(2)² + 3



f(2) = 16 - 4(4) + 3



f(2) = 16 - 16 + 3



f(2) = 3

For x = 0:





f(0) = (0)⁴ - 4(0)² + 3



f(0) = 0 - 0 + 3



f(0) = 3

```python
# Polynomial function examples
def cubic_function(x):
    return x**3 - 3*x**2 + 2*x + 1

def quartic_function(x):
    return x**4 - 4*x**2 + 3
```

![Cubic and Quartic Functions](/content/learn/math/functions/cubic-quartic-functions.png)

Just look at it - it seems interesting, no need to master it yet.

### 3. Exponential Functions

Functions with constant base raised to variable power: **f(x) = aˣ**

```python
# Exponential function examples
def exponential_function(x):
    return 2**x

def exponential_e(x):
    return np.exp(x)
```

![Exponential Functions](/content/learn/math/functions/exponential-functions.png)

Careful! The y axis is exponential.

If we make it linear, it looks like this:

![Exponential Functions Linear Scale](/content/learn/math/functions/exponential-functions-log-scale.png)





### 4. Trigonometric Functions

Functions based on angles and periodic behavior

```python
# Trigonometric function examples
def sine_function(x):
    return np.sin(x)

def cosine_function(x):
    return np.cos(x)
```

![Trigonometric Functions](/content/learn/math/functions/trigonometric-functions.png)

This is used in Rotory Positional Embeddings (RoPE) - LLM is using it to know the order of words (tokens) in the text.







Functions are using in neural networks a lot: forward propagation, backward propagation, attention, activation functions, gradients, and many more.

You don't need to learn them yet, just check them out.

### 1. Sigmoid Function

![Sigmoid Formula](/content/learn/math/functions/sigmoid-formula.png)

**e** is a famous constant (Euler's number) used in math everywhere, it's value is approximately 2.718

**f(x) = 1 / (1 + e^(-x))**

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

![Sigmoid Function and Derivative](/content/learn/math/functions/sigmoid-function-derivative.png)

We will learn derivativers in the next lesson, but I included the images here - derivative tells you how fast the function is changing - you see that when sigmoid function is growing fastest (in the middle), the derivative value is spiking.

Just look at the slope of the function, if it's big (changing fast), the derivative will be big.

### 2. ReLU (Rectified Linear Unit)

**f(x) = max(0, x)**

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

![ReLU Function and Derivative](/content/learn/math/functions/relu-function-derivative.png)

### 3. Tanh Function

![Tanh Formula](/content/learn/math/functions/tanh-formula.png)

**f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))**

```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

![Tanh Function and Derivative](/content/learn/math/functions/tanh-function-derivative.png)

**Congratulations! You finished functions for neural networks lesson!**