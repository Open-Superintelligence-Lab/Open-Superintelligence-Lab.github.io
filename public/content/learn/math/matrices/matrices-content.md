---
hero:
  title: "Matrices"
  subtitle: "Operations and Transformations for Neural Networks"
  tags:
    - "📐 Mathematics"
    - "⏱️ 12 min read"
---

**[video coming soon]**

**Level:** Beginner → Intermediate.

---

## 1. What is a matrix?

A matrix is a rectangular array of numbers arranged in rows and columns. We write an `(m x n)` matrix as:









`(m)` is the number of rows, `(n)` the number of columns.

If `(m=n)` the matrix is **square**.

**Why matrices?** They represent neural network weights, linear transformations, systems of linear equations, data tables, graphs, and more.



## 2. Notation and basic examples

**Entries:** `(A_ij)` is element in row `(i)`, column `(j)`.

**Row vector:** 1×n, **column vector:** m×1.

### Example matrices

We will use these 2 matrices below.



## 3. Step-by-step matrix operations

### 3.1 Addition and subtraction (elementwise)

Only for matrices of the same size. Add corresponding elements.

**Example:** `(A+B)`

### 3.2 Scalar multiplication

Multiply each element by the scalar. For `(2A)`:





### 3.3 Matrix multiplication

You do a dot product of a row of th first matrix with the column of the second matrix and write result at the position where that row and column intercept.

If `(A)` is `(m x p)` and `(B)` is `(p x n)`, then `(AB)` is `(m x n)`. Multiply rows of `(A)` by columns of `(B)` and sum.

**Example:** multiply the two 2×2 matrices above.

**Important:** Matrix multiplication is generally **not commutative**: `(AB is not equal to BA)` in general.



## 4. Key matrix transformations and properties

### 4.1 Transpose

### 4.2 Determinant (square matrices)

### 4.3 Inverse (when it exists)

### 4.4 Rank

The **rank** is the dimension of the column space (or row space). If rank = n for an `(n x n)` matrix, it's **full rank** and **invertible**.

### 4.5 Special matrices (common types)







## 5. Common pitfalls and tips

- Remember matrix multiplication order matters.

- Watch dimensions carefully (rows of left must equal columns of right).

- Numerical stability: beware near-singular matrices (determinant ≈ 0).