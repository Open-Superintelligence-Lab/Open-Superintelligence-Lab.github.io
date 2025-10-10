---
hero:
  title: "Understanding Derivatives"
  subtitle: "The Foundation of Neural Network Training"
  tags:
    - "📐 Mathematics"
    - "⏱️ 10 min read"
---

[video coming soon]

What are Derivatives?

A derivative measures how a function changes as its input changes.

Intuitive Understanding

Think of driving a car:





Your position is a function of time: position(t)



Your speed is the derivative of position: speed = d(position)/dt



Speed tells you how fast your position is changing

If x goes from 3 to 4, does f(x), that is y, change fast, eg. 6 to 40 or slower, eg. 6 to 7

Derivative tells us the instantaneous rate of change of a function at any point.

Mathematical Definition

The derivative of f(x) at point x is:

f'(x) = lim[h→0] (f(x+h) - f(x)) / h

Visual Representation



Here we have linearly growing function.

Derivative is always 3 for any x value, which means that in the original function, growth of y at any point is 3x (if you increase x by 1, y will increase by 3, check it).





Here you can see that as y grows faster and faster in original function (square functions grow very fast).

Derivative shows this accelerating growth, you can notice that derivative is increasing (linearly) - which means the growth is accelerating.

In previous example derivative was always 3, which meant that function is always consistantly growing by 3 times x.

Here, on the other hand, the growth is growing. 

Common Derivative Rules

You will never calculate derivatives manually, but researcher needs to understand how it works.

1. Power Rule

If f(x) = xⁿ, then f'(x) = nxⁿ⁻¹

So just put the exponent in front of the variable (or multiply with the number in front) and reduce exponent by 1.

For f(x) = x³, derivative is f'(x) = 3x²

For f(x) = 4x³, derivative is f'(x) = 4*3x² = 12x²

Step-by-Step Examples

Example 1: f(x) = x²





Using power rule: f'(x) = 2x^(2-1) = 2x¹ = 2x



Verification: f'(x) = 2x

Example 2: f(x) = x³





Using power rule: f'(x) = 3x^(3-1) = 3x²



Verification: f'(x) = 3x²

Example 3: f(x) = x⁴





Using power rule: f'(x) = 4x^(4-1) = 4x³



Verification: f'(x) = 4x³

Example 4: f(x) = √x = x^(1/2)





Using power rule: f'(x) = (1/2)x^((1/2)-1) = (1/2)x^(-1/2) = 1/(2√x)



Verification: f'(x) = 1/(2√x)

Example 5: f(x) = 1/x = x^(-1)





Using power rule: f'(x) = (-1)x^(-1-1) = (-1)x^(-2) = -1/x²



Verification: f'(x) = -1/x²



2. Constant Multiple Rule

If f(x) = c·g(x), then f'(x) = c·g'(x)

Step-by-Step Examples

Example: f(x) = 5x²

Step 1: Identify the constant and the function





Constant: c = 5



Function: g(x) = x²

Step 2: Find g'(x)





g'(x) = 2x (using power rule)

Step 3: Apply constant multiple rule





f'(x) = c·g'(x) = 5·(2x) = 10x - I showed this in the power rule as well.

Verification:





f(x) = 5x²



f'(x) = 10x ✓

Example: f(x) = -3x³

Step 1: Identify the constant and the function





Constant: c = -3



Function: g(x) = x³

Step 2: Find g'(x)





g'(x) = 3x² (using power rule)

Step 3: Apply constant multiple rule





f'(x) = c·g'(x) = (-3)·(3x²) = -9x²

Verification:





f(x) = -3x³



f'(x) = -9x² ✓



3. Sum Rule

If f(x) = g(x) + h(x), then f'(x) = g'(x) + h'(x)

Step-by-Step Examples

Example: f(x) = x² + 3x

Step 1: Identify the functions





g(x) = x²



h(x) = 3x

Step 2: Find individual derivatives





g'(x) = 2x (power rule)



h'(x) = 3 (constant multiple rule: 3·1 = 3)

Step 3: Apply sum rule





f'(x) = g'(x) + h'(x) = 2x + 3

Verification:





f(x) = x² + 3x



f'(x) = 2x + 3 ✓

Example: f(x) = x³ + 2x² + 5x + 1

Step 1: Identify the functions





g(x) = x³



h(x) = 2x²



i(x) = 5x



j(x) = 1

Step 2: Find individual derivatives





g'(x) = 3x² (power rule)



h'(x) = 4x (constant multiple rule: 2·2x = 4x)



i'(x) = 5 (constant multiple rule: 5·1 = 5)



j'(x) = 0 (constant rule)

Step 3: Apply sum rule





f'(x) = g'(x) + h'(x) + i'(x) + j'(x) = 3x² + 4x + 5 + 0 = 3x² + 4x + 5

Verification:





f(x) = x³ + 2x² + 5x + 1



f'(x) = 3x² + 4x + 5 ✓



4. Product Rule

If f(x) = g(x)·h(x), then f'(x) = g'(x)·h(x) + g(x)·h'(x)

Step-by-Step Examples

Example: f(x) = x²(x + 1)

Step 1: Identify the functions





g(x) = x²



h(x) = x + 1

Step 2: Find individual derivatives





g'(x) = 2x (power rule)



h'(x) = 1 (sum rule: derivative of x is 1, derivative of 1 is 0)

Step 3: Apply product rule





f'(x) = g'(x)·h(x) + g(x)·h'(x)



f'(x) = (2x)·(x + 1) + (x²)·(1)



f'(x) = 2x(x + 1) + x²



f'(x) = 2x² + 2x + x²



f'(x) = 3x² + 2x

Verification by expanding first:





f(x) = x²(x + 1) = x³ + x²



f'(x) = 3x² + 2x ✓

Example: f(x) = (2x + 3)(x² - 1)

Step 1: Identify the functions





g(x) = 2x + 3



h(x) = x² - 1

Step 2: Find individual derivatives





g'(x) = 2 (sum rule: derivative of 2x is 2, derivative of 3 is 0)



h'(x) = 2x (sum rule: derivative of x² is 2x, derivative of -1 is 0)

Step 3: Apply product rule





f'(x) = g'(x)·h(x) + g(x)·h'(x)



f'(x) = (2)·(x² - 1) + (2x + 3)·(2x)



f'(x) = 2(x² - 1) + (2x + 3)(2x)



f'(x) = 2x² - 2 + 4x² + 6x



f'(x) = 6x² + 6x - 2



5. Chain Rule

If f(x) = g(h(x)), then f'(x) = g'(h(x))·h'(x)

Step-by-Step Examples

Example: f(x) = (x² + 1)³

Step 1: Identify the inner and outer functions





Inner function: h(x) = x² + 1



Outer function: g(u) = u³ (where u = h(x))

Step 2: Find individual derivatives





h'(x) = 2x (sum rule: derivative of x² is 2x, derivative of 1 is 0)



g'(u) = 3u² (power rule)

Step 3: Apply chain rule





f'(x) = g'(h(x))·h'(x)



f'(x) = 3(h(x))²·(2x)



f'(x) = 3(x² + 1)²·(2x)



f'(x) = 6x(x² + 1)²

Verification by expanding first:





f(x) = (x² + 1)³ = (x² + 1)(x² + 1)(x² + 1)



Expanding: f(x) = x⁶ + 3x⁴ + 3x² + 1



f'(x) = 6x⁵ + 12x³ + 6x = 6x(x⁴ + 2x² + 1) = 6x(x² + 1)² ✓

Example: f(x) = √(x² + 4)

Step 1: Identify the inner and outer functions





Inner function: h(x) = x² + 4



Outer function: g(u) = √u = u^(1/2) (where u = h(x))

Step 2: Find individual derivatives





h'(x) = 2x (sum rule: derivative of x² is 2x, derivative of 4 is 0)



g'(u) = (1/2)u^(-1/2) = 1/(2√u) (power rule)

Step 3: Apply chain rule





f'(x) = g'(h(x))·h'(x)



f'(x) = (1/(2√(x² + 4)))·(2x)



f'(x) = 2x/(2√(x² + 4))



f'(x) = x/√(x² + 4)



Derivatives of Neural Network Functions

1. Sigmoid Function





f(x) = 1 / (1 + e^(-x))

Step-by-Step Derivative Calculation

To find the derivative of sigmoid, we'll use the quotient rule and chain rule.

Usually you will ChatGPT sigmoid derivative, but let's see how it's derived.

Step 1: Rewrite the function





f(x) = 1 / (1 + e^(-x))



Let u = 1 + e^(-x), so f(x) = 1/u

Step 2: Apply quotient rule





f'(x) = (0·u - 1·u') / u² = -u' / u²

Step 3: Find u' using chain rule





u = 1 + e^(-x)



u' = 0 + e^(-x) · (-1) = -e^(-x)

Step 4: Substitute back





f'(x) = -(-e^(-x)) / (1 + e^(-x))²



f'(x) = e^(-x) / (1 + e^(-x))²

Step 5: Simplify





f'(x) = e^(-x) / (1 + e^(-x))²



f'(x) = [e^(-x) / (1 + e^(-x))] · [1 / (1 + e^(-x))]



f'(x) = [1 / (1 + e^(-x))] · [e^(-x) / (1 + e^(-x))]



f'(x) = f(x) · [e^(-x) / (1 + e^(-x))]

Step 6: Further simplification





Notice that e^(-x) / (1 + e^(-x)) = 1 - 1/(1 + e^(-x)) = 1 - f(x)



Therefore: f'(x) = f(x) · (1 - f(x))

Final Result: f'(x) = f(x)(1 - f(x))



Chain Rule

Chain rule is how neural networks learn (backpropagation).

Mathematical Statement

If y = f(g(x)), then dy/dx = (dy/dg) × (dg/dx)

Neural Network Application

In neural networks, we often have functions like: f(x) = activation(linear_transformation(x))

Step-by-Step Chain Rule Example

Example: Neural Network Layer with Sigmoid Activation

Given:





Linear transformation: z = 2x + 1



Activation function: σ(z) = 1/(1 + e^(-z))



Composite function: f(x) = σ(2x + 1)

Step 1: Identify inner and outer functions





Inner function: h(x) = 2x + 1



Outer function: g(z) = σ(z) = 1/(1 + e^(-z))

Step 2: Find individual derivatives





h'(x) = 2 (derivative of 2x + 1)



g'(z) = σ(z)(1 - σ(z)) (sigmoid derivative)

Step 3: Apply chain rule





f'(x) = g'(h(x)) · h'(x)



f'(x) = σ(2x + 1)(1 - σ(2x + 1)) · 2



f'(x) = 2σ(2x + 1)(1 - σ(2x + 1))

Step 4: Calculate at specific point (x = 1)

Step 4a: Calculate h(1)





h(1) = 2(1) + 1 = 3

Step 4b: Calculate σ(3)





σ(3) = 1/(1 + e^(-3)) = 1/(1 + 0.050) = 1/1.050 ≈ 0.953

Step 4c: Calculate σ'(3)





σ'(3) = σ(3)(1 - σ(3)) = 0.953(1 - 0.953) = 0.953(0.047) ≈ 0.045

Step 4d: Apply chain rule





f'(1) = σ'(3) · h'(1) = 0.045 · 2 = 0.090

Final Answer: f'(1) ≈ 0.090



Partial Derivatives

When we have functions of multiple variables, we use partial derivatives.

Definition

For f(x, y), the partial derivative with respect to x is: ∂f/∂x = lim[h→0] (f(x+h, y) - f(x, y)) / h

Example: Linear Function

f(x, y) = 2x + 3y + 1

Step-by-Step Partial Derivative Calculation

Finding ∂f/∂x (partial derivative with respect to x):

Step 1: Treat y as a constant





f(x, y) = 2x + 3y + 1



When taking ∂f/∂x, we treat y as constant, so 3y + 1 is constant

Step 2: Differentiate with respect to x





∂f/∂x = ∂/∂x(2x) + ∂/∂x(3y) + ∂/∂x(1)



∂f/∂x = 2 + 0 + 0 = 2

Finding ∂f/∂y (partial derivative with respect to y):

Step 1: Treat x as a constant





f(x, y) = 2x + 3y + 1



When taking ∂f/∂y, we treat x as constant, so 2x + 1 is constant

Step 2: Differentiate with respect to y





∂f/∂y = ∂/∂y(2x) + ∂/∂y(3y) + ∂/∂y(1)



∂f/∂y = 0 + 3 + 0 = 3

Final Results:





∂f/∂x = 2



∂f/∂y = 3

Hand Calculation Examples

Example: Find partial derivatives at (x, y) = (1, 2)





∂f/∂x = 2 (constant, doesn't depend on x or y)



∂f/∂y = 3 (constant, doesn't depend on x or y)

Example: Find partial derivatives at (x, y) = (5, -1)





∂f/∂x = 2 (still constant)



∂f/∂y = 3 (still constant)



Example: Quadratic Function

f(x, y) = x² + 2xy + y²

Step-by-Step Partial Derivative Calculation

Finding ∂f/∂x (partial derivative with respect to x):

Step 1: Treat y as a constant





f(x, y) = x² + 2xy + y²



When taking ∂f/∂x, we treat y as constant

Step 2: Differentiate with respect to x





∂f/∂x = ∂/∂x(x²) + ∂/∂x(2xy) + ∂/∂x(y²)



∂f/∂x = 2x + 2y + 0 = 2x + 2y

Finding ∂f/∂y (partial derivative with respect to y):

Step 1: Treat x as a constant





f(x, y) = x² + 2xy + y²



When taking ∂f/∂y, we treat x as constant

Step 2: Differentiate with respect to y





∂f/∂y = ∂/∂y(x²) + ∂/∂y(2xy) + ∂/∂y(y²)



∂f/∂y = 0 + 2x + 2y = 2x + 2y

Final Results:





∂f/∂x = 2x + 2y



∂f/∂y = 2x + 2y

Hand Calculation Examples

Example: Find partial derivatives at (x, y) = (1, 2)

Step 1: Calculate ∂f/∂x





∂f/∂x = 2(1) + 2(2) = 2 + 4 = 6

Step 2: Calculate ∂f/∂y





∂f/∂y = 2(1) + 2(2) = 2 + 4 = 6

Example: Find partial derivatives at (x, y) = (3, -1)

Step 1: Calculate ∂f/∂x





∂f/∂x = 2(3) + 2(-1) = 6 - 2 = 4

Step 2: Calculate ∂f/∂y





∂f/∂y = 2(3) + 2(-1) = 6 - 2 = 4