**[video comingn soon]**

Welcome! This guide will introduce you to vectors, which are fundamental objects in mathematics, physics, and computer science. We'll explore what they are and how to work with them, focusing on the concepts, not the code.

---

## Step 1: What is a Vector?

At its core, a **vector** is a mathematical object that has both **magnitude** (length or size) and **direction**.





Think about the difference between "speed" and "velocity."

- **Speed** is a single number (a scalar), like 50 km/h. It only tells you the magnitude.

- **Velocity** is a vector, like 50 km/h north. It tells you both the magnitude (50 km/h) and the direction (north).

We represent vectors as a list of numbers called **components**. For example, in a 2D plane, a vector `v` can be written as:

```
v = [x, y]
```

This notation means "start at the origin (0,0), move x units along the horizontal axis, and y units along the vertical axis." The arrow drawn from the origin to that point (x, y) is the vector.

**Examples:**

- `v = [3, 4]` represents an arrow pointing to the coordinate (3, 4).

- `u = [-2, 1]` represents an arrow pointing to the coordinate (-2, 1).



## Step 2: The Two Core Properties: Magnitude and Direction

Every vector is defined by these two properties.

### Magnitude (Length)

The **magnitude** of a vector is its length. It's often written with double bars, like `||v||`. We can calculate it using the Pythagorean theorem. For a 2D vector `v = [x, y]`, the formula is:

```
||v|| = √(x² + y²)
```

For a 3D vector `w = [x, y, z]`, it's a natural extension: `||w|| = √(x² + y² + z²)`.

**Example:**
For `v = [3, 4]`:
```
||v|| = √(3² + 4²) = √(9 + 16) = √25 = 5
```
The length of the vector [3, 4] is 5 units.

### Direction (Unit Vectors)

How can we describe only the direction of a vector, ignoring its length? We use a **unit vector**. A unit vector is any vector that has a magnitude of exactly 1.

To find the unit vector of any given vector, you simply divide the vector by its own magnitude. This scales the vector down to a length of 1 while preserving its direction. The unit vector is often denoted with a "hat," like `û`.

```
û = v / ||v||
```

**Example:**
For `v = [3, 4]`, we know `||v|| = 5`.
The unit vector `û` is:
```
û = [3, 4] / 5 = [3/5, 4/5] = [0.6, 0.8]
```
This new vector [0.6, 0.8] points in the exact same direction as [3, 4], but its length is 1.



## Step 3: Vector Arithmetic

We can perform operations on vectors to combine or modify them.

### Vector Addition

**Geometrically**, adding two vectors `u + v` means placing the tail of vector `v` at the tip of vector `u`. The resulting vector, `w`, is the arrow drawn from the original starting point to the tip of the second vector.

**Mathematically**, we just add the corresponding components:
If `u = [x₁, y₁]` and `v = [x₂, y₂]`, then:
```
u + v = [x₁ + x₂, y₁ + y₂]
```

### Scalar Multiplication

Multiplying a vector by a regular number (a **scalar**) changes its magnitude but not its direction (unless the scalar is negative, in which case the direction is reversed).

If `k` is a scalar and `v = [x, y]`, then:
```
k  v = [kx, k*y]
```


**Examples:**

- `2 * v` doubles the vector's length.

- `0.5 * v` halves the vector's length.

- `-1 * v` flips the vector to point in the opposite direction.







## Step 4: The Dot Product

The **dot product** is a way of multiplying two vectors that results in a single number (a scalar). It is one of the most important vector operations.

**Intuition:** The dot product tells you how much two vectors align or point in the same direction.

- **Large positive dot product:** The vectors point in very similar directions.

- **Dot product is zero:** The vectors are perpendicular (orthogonal) to each other.

- **Large negative dot product:** The vectors point in generally opposite directions.

**Calculation:** To calculate the dot product, you multiply the corresponding components and then add the results.
If `u = [x₁, y₁]` and `v = [x₂, y₂]`, the dot product `u · v` is:

```
u · v = (x₁  x₂) + (y₁  y₂)
```

### Geometric Meaning & Finding Angles
The dot product also has a powerful geometric definition:

```
u · v = ||u||  ||v||  cos(θ)
```

where `θ` (theta) is the angle between the two vectors. We can rearrange this formula to find the angle between any two vectors!

```
cos(θ) = (u · v) / (||u|| * ||v||)
```

This is an incredibly useful property, allowing us to calculate angles in any number of dimensions.

## Step 5: Neural Networks:

Every input, hidden state, and output is a vector.

- A single image, sound, or sentence is converted into a vector of numbers that captures its features.

- Each neuron operates on these vectors — combining them through dot products, matrix multiplications, and nonlinear activations to extract patterns.

- When you train a neural network, you're really adjusting weight vectors so that the model transforms input vectors into desired output vectors.



### 💬 In Large Language Models (LLMs):

LLMs represent words, sentences, and even abstract concepts as high-dimensional vectors (embeddings).

- The vector for a word like "king" is close to "queen" in this space because their meanings are similar.

- Attention mechanisms compute dot products between vectors to measure how related words are in context — that's how the model "focuses" on relevant information.

- The entire reasoning process of an LLM — understanding, summarizing, generating — happens through transformations of these vectors.

**By understanding vectors, you understand how neural networks think, learn, and represent meaning.**