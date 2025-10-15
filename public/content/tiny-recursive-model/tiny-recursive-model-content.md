---
hero:
  title: "Tiny Recursive Model"
  subtitle: "New recursive reasoning AI architecture"
  tags:
    - "⏱️ Technical Deep Dive"
    - "📄 Research Article"
---

## The New AI Reasoning Architecture 

#### How a 7M Model Beats 1T Models at Sudoku, Mazes, ARC-AGI

**The Tiny Reasoning Model (TRM)** uses a 2-layer transformer (7M parameters) that reuses the same layers hundreds of times to reason about problems recursively.

It beats 100x bigger models in Sudoku-Extreme, Mazes, ARC-AGI and more.

In this tutorial we will learn how TRM works and do our own experiments.

---

## TRM Architecture Overview

![Tiny Recursive Model Architecture](/content/tiny-recursive-model/images/tiny-recursive-model-architecture.png)
*Figure: The Tiny Recursive Model architecture showing the main processing block (4x transformer layers), input combination of question (x), answer (y), and reasoning (z), output processing for loss calculation, and the recursive update mechanism that iteratively refines the reasoning and prediction over up to 16 steps.*

The diagram above illustrates the complete TRM architecture. The model processes three key components:
- **Input (x)**: The question or problem to solve (e.g., maze layout)
- **Prediction (y)**: The model's current answer attempt
- **Latent (z)**: The model's internal reasoning state

These are combined and processed through a 4-layer transformer stack, with the output used to compute cross-entropy loss. The key innovation is the recursive update mechanism at the bottom, which iteratively refines both the reasoning (z) and prediction (y) over multiple steps to progressively improve the solution.

---

## How TRM Works

### Step 1: Setup

Let's train TRM to solve a maze.

**1. Representing the Maze as a Grid:**
First, we show maze as a grid of numbers. Each cell in the grid gets a number.

-   `0` = Empty path
-   `1` = Wall
-   `2` = Start point
-   `3` = End point

For a concrete example, let's trace a tiny 3x3 maze.

-   **`x_input`** (The unsolved maze)
    ```
    [[2, 0, 1],
     [1, 0, 1],
     [1, 0, 3]]
    ```
-   **`y_true`** (The correct solution, with `4` representing the path)
    ```
    [[2, 4, 1],
     [1, 4, 1],
     [1, 4, 3]]
    ```

**2. Tokenization:**
The term **token** just means a single unit of our data. In this case, a single number in the grid (`0`, `1`, `2`, `3`, or `4`) is a token. To make it easier for the network to process, we "unroll" the grid into a long 1D list.

For our 3x3 example, the grid is unrolled into a list of 9 tokens.

**3. Embedding: Giving Meaning to Numbers:**
To let the model understand what numbers like `4` and `1` mean, we will asign a big **vector embedding** to each. Vector embedding is a long vector (array of numbers) that the model can modify to store information about the wall, empty path, etc.

These vectors will represent the meaning of a "wall" or "end point".

I recommend you remind yourself what vector embeddings (in LLMs, of words, tokens, etc) are by searching it on YouTube or talking to an AI chatbot.

-   An **Embedding Layer** is like a dictionary.
-   It contains vector embeddings for each of our numbers.
-   `1`: `[0.3, -1.2, 0.7, 0.0, 1.5, -0.4, 0.9, 2.3]`  ←  Example vector embedding for "wall"
-   **Output:** A long list of numbers called a **vector**. This vector represents the *meaning* of "wall" in a way the network can understand. The network itself choose (learned) numbers within this vector during the training so it can "understand" it.

After this step, our input maze is no longer a list of simple numbers. It's a list of vectors. For our 3x3 maze, if we use a vector of size 8 for each token, our input becomes:

-   `x`: A `9x8` matrix of vectors representing the maze.

This rich representation is what we feed to the main model.

---

### Step 2: The Core Architecture: The TRM Brain

The "brain" of TRM is a tiny 2-layer transformer called `net`. It processes information to produce an output. To "think," TRM uses two variables, both having same shape as `x`:

-   `y`: The model's current **best guess** for the solution. Might be wrong"
```
[[2, 4, 1],
  [1, 4, 1],
  [1, 0, 3]]
```
-   `z`: A **latent thought**. `z` tells what needs to be changed in `y` to turn it into a correct solution. `z` is passed through the transformer multiple times to let the model refine what needs to be changed in `y`, this is how the model reasons or thinks. Then the change is applied to `y`.

For our 3x3 example, `z` and `y` start as `9x8` matrices of zeros.

---

### Step 3: The Learning Process, from the Inside Out

TRM learns in a series of nested loops. Let's start from the core and build our way out.

#### The Innermost Loop: `latent_recursion` (The Core Thought)

This is where the tiny `net` (a 2-layer Transformer) does all its work. The process is broken into two phases that repeat to form a cycle of thinking and refining.

**Phase A: Reasoning (Updating the Scratchpad `z`)**
The model "thinks" by refining its internal planning token `z`, in a loop of 6 steps. The goal is to build a better and better plan for changing `y`.

1.  **The Process:** In each of the 6 steps, the `net` takes three inputs:
    -   The maze itself (`x`).
    -   The model's current best guess for the solution (`y`) - this could be all zeroes at the beginning.
    -   The scratchpad from the previous step (`z`).
2.  **How it works:**
    -   **Combining Inputs:** The three inputs are added together element-wise (`x + y + z`). This creates a single sequence of rich vectors, where each vector (representing a cell in the maze) contains combined information about the maze layout (`x`), the current guess (`y`), and the ongoing thought process (`z`).
    -   **Thinking with Attention:** This combined sequence is fed into the 2-layer Transformer. The Transformer's self-attention mechanism allows it to look at all the cells at once and identify relationships. For example, it can see how the "start" cell relates to a potential path cell, informed by the input data `x` and the reasoning `z`.
    -   **Generating the Next Thought:** The two transformer layers process this information and output a new sequence of vectors of the exact same shape. This output *is* the new `z`. There isn't a separate "output head" to generate it; the transformation performed by the two layers *is* the act of creating the next, more refined thought. Even though the input was a sum containing `x` and `y`, the network learns to produce an output that serves as a useful new `z` for the next step.

    This process repeats 6 times, meaning the information is passed through the same two layers six consecutive times, becoming progressively more sophisticated with each pass.
3.  **Example Trace:** After a few passes through transformer, `z` might encode low-level features like wall locations. By the sixth pass, it might represent a high-level plan for updating answer (`y`).
   
   - Interestingly, the same 2 transformer layers are used for detecting low level features, making high level plan and later for updating `y` itself. These 2 layers have multiple purposes, which is the power of neural networks, it can learn to do multiple, less related or unrelated transformations taht only depends on the input data.

**Phase B: Refining the Answer (Updating the Guess `y`)**
After the 6-step reasoning loop, using the latest latent though `z` the model updates its answer `y`.

-   **How it works:** It combines its previous answer (`y`) with its final, refined thought (`z`) by adding them together (`y + z`) and passes the result through the same `net` one last time. The output is the new, improved `y`.
    -   **Crucially, `x` is not included in this step.** This is a deliberate design choice that tells the single `net` which task to perform.
    -   `x` is present in reasoning (`x + y + z`).
    -   `x` is absent in answer refinement (`y + z`).

The reason I said "answer refinement" is because this 6+1 loop happens multiple times, each time "thinking" for 6 passes and updating `y` once.

#### The Middle Loop: `deep_recursion` (The Full Thought Process)

Now that we understand how reasoning + y refinement loop works, let's see the full thought process from the beginning where this whole loop is repeated 3 times to get the best `y`.

The previously described inner loop (the 6+1 steps of reasoning and `y` refinement) runs `T` times (e.g., `T=3`). The state (`y` and `z`) is **carried over** between these runs; it is not reset to zero.

-   **Round 1 (Warm-up):** Starts with a blank (all zeroes) `y` and `z` (remember, this is the absolute beginning of the process, so there is no `y` and `z` to carry over). It runs the full inner loop (6 reasoning + 1 `y` refinement steps) to produce smarter `y_1` and `z_1`. This is done in "no-gradient" mode for speed and memory savings - neural network doesn't learn here.
-   **Round 2 (Warm-up):** It takes `y_1` and `z_1` as its starting point and runs the inner loop again to produce an even better `y_2` and `z_2`. Still no gradients and learning.
-   **Round 3 (For Real):** It starts with the well-reasoned `y_2` and `z_2`, runs the inner loop one final time, and this time all calculations are tracked so the model can learn with backpropagation.

This process of warming up the model's "thought" before the final, learnable step is a key optimization.

#### The Outermost Loop: Even more loops!

The model gets multiple "chances" (up to 16) to solve the same maze, and after each chance, it refines its `net` weights. The state (`y` and `z`) **is carried over** from one middle loop iteration to the next, as shown in the paper's pseudocode. It allows the model to get multiple "chances" (up to 16) to solve the same maze, improving with each one.

This is just repeating middle loop up to 16 times. Model can decide to stop earlier than 16 if it feels like it got the correct answer.

Why we need this loop:

After each middle loop itteration this outter loop updates weights once (remember that the Round 3 in middle loop does backpropagation).

Then in the next iteration it repeats the middle loop with the updated weights, allowing the model to progressively improve its solution with each attempt.

#### Knowing when to stop thinking (The Q head)

The outer loop can run up to 16 times, but it doesn't have to. It would be a waste of time to keep thinking about a maze it has already solved.

So, the model has a little side-brain called a "Q head". After each full thought process (each middle loop), this Q head spits out a score. This score is basically the model's confidence: "How sure am I that I got this right?"

If the confidence score is high enough, the outer loop just stops (`break`), and the model moves on to the next maze.

It learns to get this confidence score right because it's part of the training. It gets rewarded if it's confident *and* correct, and penalized if it's confident but wrong. The paper calls this Adaptive Computation Time (ACT).

---

```python
# Initialize
y, z = zeros_like(x), zeros_like(x)

# Deep supervision loop (up to 16 times)
for supervision_step in range(16):
    
    # Deep recursion: warm-up (2 times, no gradients)
    with torch.no_grad():
        for _ in range(2):
            # Latent recursion
            for _ in range(6):
                z = net(x + y + z)
            y = net(y + z)
    
    # Deep recursion: final (1 time, WITH gradients)
    for _ in range(6):
        z = net(x + y + z)
    y = net(y + z)
    
    # Learn
    y_pred = output_head(y)
    loss = cross_entropy(y_pred, y_true)
    loss.backward()
    optimizer.step()
    
    # Should we stop?
    q = Q_head(y)
    if q > 0:
        break
```

---

### Step 4: Ablation Studies - What Makes TRM Work?

![Complete Ablation Study](/content/tiny-recursive-model/images/complete_ablation_study.png)
*Figure: Training loss comparison across four TRM configurations over 10 epochs on maze-solving (30x30, hard). The baseline (blue solid) uses TRM's standard design: 2-layer network, H=3 (middle loop), L=6 (inner loop), with EMA. Ablations test: removing EMA (red dashed), reducing recursion depth (green dash-dot), and using a bigger 4-layer network (magenta dotted).*

To understand what makes TRM effective, we systematically test variations by removing or changing key components. These **ablation studies** reveal which design choices are essential.

#### Experimental Setup

We test four configurations on a maze-solving task (30x30 hard mazes, 1000 training examples):

| Configuration | Layers | H_cycles | L_cycles | EMA | Effective Depth* |
|---------------|--------|----------|----------|-----|------------------|
| **Baseline TRM** | 2 | 3 | 6 | Yes | 42 |
| **No EMA** | 2 | 3 | 6 | No | 42 |
| **Less Recursion** | 2 | 2 | 2 | Yes | 12 |
| **Bigger Brain** | 4 | 3 | 3 | Yes | 48 |

*Effective depth = T × (n+1) × layers

#### Results

**Note:** These are 10-epoch experiments—a very small amount of training compared to the paper's 50,000+ epoch runs. Longer training may significantly change the relative performance of these configurations, particularly for generalization (as we see with the "Bigger Brain" results below).

| Configuration | Initial Loss | Final Loss | Min Loss | Improvement |
|---------------|--------------|------------|----------|-------------|
| Baseline | 1.789 | 1.062 | 1.045 | 40.6% |
| No EMA | 1.789 | 1.042 | 1.041 | 41.7% |
| Less Recursion | **2.100** | 1.100 | 1.042 | 47.6% |
| Bigger Brain (4-layer) | 1.789 | **1.007** | **1.007** | **43.7%** |

#### Key Findings

**1. The "Bigger Brain" Paradox: Short-Term vs. Long-Term Performance**

The 4-layer network achieved the **best final loss** (1.007) in our 10-epoch experiments, outperforming the 2-layer baseline by ~5%. This seems to contradict the paper's claim that "less is more."

**Why the Difference?**
- **Short-term** (10 epochs): More capacity = faster learning. The 4-layer network could be memorizing patterns quickly.
- **Long-term** (50k+ epochs): More capacity = overfitting. The 2-layer network is *forced* to learn reusable reasoning strategies instead of memorizing specific solutions.
  
Paper's core thesis: **Small networks forced to think recursively generalize better than large networks**, even if they train slower initially. The 2-layer architecture is chosen specifically to prevent memorization and force reliance on recursion.

**2. Recursion Depth is Fundamental**

The "Less Recursion" configuration (H=2, L=2) shows severely degraded performance:
- Started at **17% higher initial loss** (2.100 vs 1.789) before any training
- Achieved worst final loss (1.100) despite improving 47.6%

**What the Paper Says:** Reducing recursion from T=3, n=6 to T=2, n=2 drops Sudoku accuracy from 87.4% to 73.7% — a 14% decline.

**Why This Matters:** The high initial loss reveals that shallow recursion cripples the model's representational power *by design*. Even with perfect training, there aren't enough recursive "thinking steps" to solve complex problems. **You cannot compensate for insufficient recursion depth with better training.**

**3. EMA Has Minimal Short-Term Impact**

Removing EMA barely affected 10-epoch performance (final loss 1.042 vs 1.062 for baseline, only ~2% difference).

**What the Paper Says:** On Sudoku-Extreme, removing EMA drops accuracy from 87.4% to 79.9% — an 8% decline after full training.

**Why the Difference?** EMA is an **Exponential Moving Average** of model weights that stabilizes training over long runs. In short experiments, both models are still exploring and haven't yet encountered the instability that EMA prevents. Over 50,000+ epochs, EMA prevents catastrophic divergence and overfitting spikes, making it essential for final performance.

---

Thank you for reading this tutorial and see you in the next one.