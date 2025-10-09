---
hero:
  title: "Understanding Derivatives"
  subtitle: "The Foundation of Neural Network Training"
  tags:
    - "📐 Mathematics"
    - "⏱️ 10 min read"
---

# Understanding Derivatives

## What is a Derivative?

A derivative measures how a function changes as its input changes. In simpler terms, it tells us the **rate of change** or the **slope** of a function at any given point.

The formal definition is:

```
f'(x) = lim(h→0) [f(x+h) - f(x)] / h
```

Think of it this way: if you're driving a car, your speed at any moment is the derivative of your position with respect to time. It tells you how quickly your position is changing.

![Derivative Visualization](derivative-graph.png)

## Why Derivatives Matter in AI

Derivatives are the **backbone of neural network training**. They allow us to:

- **Calculate gradients** for backpropagation
- **Optimize model parameters** to minimize loss
- **Understand how changes** in weights affect the output
- **Implement gradient descent** algorithms effectively

Without derivatives, we couldn't train modern neural networks!

## Common Derivatives in Neural Networks

Here are the most important derivative rules you'll encounter:

### Power Rule
```
d/dx(x^n) = n·x^(n-1)
```

**Example**: If f(x) = x³, then f'(x) = 3x²

### Chain Rule
```
d/dx[f(g(x))] = f'(g(x))·g'(x)
```

This is **crucial** for backpropagation through multiple layers!

### Exponential Function
```
d/dx(e^x) = e^x
```

### Natural Logarithm
```
d/dx(ln(x)) = 1/x
```

### Sum Rule
```
d/dx[f(x) + g(x)] = f'(x) + g'(x)
```

### Product Rule
```
d/dx[f(x)·g(x)] = f'(x)·g(x) + f(x)·g'(x)
```

## Practical Example: Loss Function

Let's say we have a simple loss function:

```
L(w) = w²
```

The derivative is:
```
dL/dw = 2w
```

**What does this tell us?**

- If w = 3, then dL/dw = 6 (loss increases rapidly as w increases)
- If w = -2, then dL/dw = -4 (loss decreases as w increases)
- If w = 0, then dL/dw = 0 (we're at the minimum!)

We use this information to **update our weights** during training:

```python
# Gradient descent update
w_new = w_old - learning_rate * dL/dw
```

## Visualizing the Derivative

![Tangent Line](tangent-line.png)

The derivative at a point gives us the **slope of the tangent line** at that point. This visual intuition is powerful when understanding how neural networks learn.

## Key Takeaways

✅ Derivatives measure rates of change  
✅ Chain rule enables backpropagation through deep networks  
✅ Gradients point in the direction of steepest increase  
✅ We move in the opposite direction to minimize loss  
✅ Understanding derivatives is essential for debugging and improving neural networks

## Next Steps

Now that you understand derivatives, you're ready to explore **functions** and how they transform data in neural networks!

