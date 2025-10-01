# DeepSeek Sparse Attention

Prerequisites: Attention Mechanism

**📺 Recommended Video Resource:** For a comprehensive understanding of attention mechanisms and DeepSeek's Multihead Latent Attention, watch this video: [https://youtu.be/TfEG0TwueTs](https://youtu.be/TfEG0TwueTs)

*   **If you're new to attention mechanisms:** Start from the beginning of the video
*   **If you want to focus on DeepSeek's Multihead Latent Attention (MLA):** Jump to 38:53 or use this direct link: [https://youtu.be/TfEG0TwueTs?t=2333](https://youtu.be/TfEG0TwueTs?t=2333)
*   **Note:** I will explain MLA again in this article / video, but I recommend watching both for better understanding.

Standard Transformers use an "attention" mechanism where every new token being generated looks back at all the previous tokens in the sequence.

This is computationally very expensive. If you have a sequence of length L, the complexity is O(L²), meaning the computation and memory required grow quadratically.

Doubling the text length from 10,000 to 20,000 tokens doesn't just double the cost—it quadruples it. This makes processing very long documents (like books or large codebases) prohibitively slow and expensive.

Instead of having each token attend to all previous tokens, DeepSeek Sparse Attention (DSA) intelligently selects a small, fixed-size subset (k) of the most relevant previous tokens to attend to. This changes the complexity from O(L²) to O(L * k), which is much more manageable since k is a small constant (e.g., 2048) and L can be very large (e.g., 128,000).


DSA is made of two main components:

The lightning indexer will perform full attention between every token but it's a lot smaller and faster attenion - ReLU actionvation which is very fast and a lot smaller dimension of key and query.

#### Component 1: The Lightning Indexer

This is a fast and lightweight mechanism whose only job is to figure out which past tokens are important for the current token.

*   **How it works:** For the current token (`h_t`), the indexer quickly calculates an "index score" (`I_t,s`) for every previous token (`h_s`). This score represents the predicted relevance of token `s` to token `t`.
*   **Formula `(1)`:** The formula 1 is essentially a simplified attention calculation. It uses its own small set of queries (`q_I`) and keys (`k_I`) to compute these scores.
*   **Why it's "Lightning":** It's designed for speed. It uses a simple `ReLU` activation function and can be run with low-precision numbers (FP8), making it computationally very cheap, even though it still technically looks at all previous tokens (an `O(L²)` operation, but a very, very fast one).

### 1. The Formulas Explained (The "What")

The paper provides two key formulas that describe this two-step process.

#### **Formula (1): The Lightning Indexer**

`I_t,s = Σ H_I (j=1) [w_t,j^I ⋅ ReLU(q_t,j^I ⋅ k_s^I)]`

This formula calculates the **index score** (`I_t,s`), which represents the "relevance" of a past token `s` to the current token `t`. Let's break it down:

*   `I_t,s`: The final importance score. A higher score means token `s` is more important for token `t`.
*   `h_t` and `h_s`: These are the vector representations (hidden states) of the current token (`t`) and a previous token (`s`).
*   `q_t,j^I` and `k_s^I`: These are special, lightweight **query** and **key** vectors created just for the indexer (indicated by the `I` superscript). They are derived from `h_t` and `h_s` respectively.
*   `q_t,j^I ⋅ k_s^I`: This is a dot product, the fundamental operation in attention. It measures the similarity or compatibility between the query and the key.
*   `ReLU(...)`: A simple activation function (Rectified Linear Unit). It's very fast to compute. If the dot product is negative, it becomes 0; otherwise, it stays the same.
*   `w_t,j^I`: An additional weight, also derived from the query token `h_t`. It acts as a learned gate or importance factor for each indexer head `j`.
*   `Σ ...`: This sums the results across all the indexer's heads (`H^I`). The indexer has only a few heads to keep it fast.

**In simple terms:** The Lightning Indexer is a mini, simplified attention mechanism. Its only job is to quickly calculate a relevance score for every pair of tokens without doing the full, expensive attention computation.

#### **Formula (2): The Main Attention Calculation**

`u_t = Attn(h_t, {c_s | I_t,s ∈ Top-k(I_t,:)})`

This formula describes how the final output (`u_t`) is computed after the selection is done.

*   `u_t`: The final output hidden state for the current token `t`.
*   `Attn(...)`: This represents the main, powerful attention mechanism (in this case, Multi-Query Attention).
*   `h_t`: The query from the current token.
*   `{c_s | I_t,s ∈ Top-k(I_t,:)}`: This is the most important part. It means: "Use the set of key-value entries `c_s` **only if** their corresponding index score `I_t,s` (calculated in Formula 1) is among the `top-k` highest scores for the current token `t`."

**In simple terms:** The main attention mechanism is told to ignore almost all previous tokens and focus *only* on the handful of key-value entries that the Lightning Indexer identified as most important.

#### Component 2: The Fine-grained Token Selection
This component is simple: it takes all the index scores calculated by the Lightning Indexer and picks the `top-k` highest scores.

*   **Function:** It acts as a gatekeeper. It tells the main, powerful attention mechanism: "You don't need to look at all 100,000 previous tokens. I've found the 2,048 most important ones for you. Just look at these."

The final attention output (`u_t`) is then calculated by the main attention module, but only using the current token's query and the `k` key-value pairs that were selected.

### Step 3: How The Model Was Trained

They didn't train this model from scratch. They cleverly adapted an existing, powerful model (**DeepSeek-V3.1-Terminus**) that was already trained on long contexts. The training happened in several stages.

#### Stage 1: Continued Pre-Training (Two Phases)

1.  **Dense Warm-up Stage:**
    *   **Goal:** To teach the brand-new Lightning Indexer what "important" tokens look like.
    *   **Method:** They froze the main model and kept the standard (dense) attention active. They then trained *only* the Lightning Indexer. The indexer's objective was to make its importance scores match the attention scores from the powerful, pre-trained main model. They used a KL-divergence loss, which is a way of measuring how similar two probability distributions are. In essence, they told the indexer: "Learn to predict what the main model *would have* paid attention to." This phase was very short (1,000 steps).

2.  **Sparse Training Stage:**
    *   **Goal:** To adapt the entire model to work with the sparse attention pattern.
    *   **Method:** They "switched on" the `top-k` selector, making the attention sparse. They unfroze the main model and trained everything together.
        *   The **main model** was trained on its usual task: predicting the next word (language modeling loss). It had to learn to perform well with only the limited context provided by the selector.
        *   The **Lightning Indexer** continued to be trained with the KL-divergence loss to align with the main model's attention, but now only on the selected `k` tokens.
    *   This was the main training phase (15,000 steps, using 943.7 billion tokens).

#### Stage 2: Post-Training
After the pre-training was done, they fine-tuned the model for specific tasks (like coding, math, reasoning, and following instructions) using Reinforcement Learning (RL). Crucially, they used the **exact same data and methods** as they did for the original DeepSeek-V3.1-Terminus model. This ensures a fair comparison between the dense and sparse models.

## Deep Dive: Multi-Head Latent Attention (MLA) Architecture

Let's break down the Multi-Head Latent Attention (MLA) architecture step-by-step, using the provided formulas and text.

The core goal of MLA is to dramatically reduce the size of the Key-Value (KV) cache, which is the main memory bottleneck when processing long sequences. It achieves this through a clever "compress-then-decompress" strategy.

The process can be split into two main parts:
1. Creating the Keys and Values (for the cache).
2. Creating the Queries (to interact with the cache).

---

### Step 1: Processing Keys and Values (Formulas 1-5)

This section explains how the model takes the input for the current token (`h_t`) and creates the Key and Value vectors that will be stored (in a compressed form) and used by future tokens.

#### Formula (1): The Compression Step
`c_t^KV = W^DKV * h_t`

*   **What it does:** This is the most critical step for saving memory. It takes the large, high-dimensional input vector for the current token (`h_t`) and projects it down into a much smaller, low-dimensional vector called the **compressed latent vector** (`c_t^KV`).
*   **`W^DKV`:** This is a learned "Down-projection" matrix. The model learns how to best squish the information from `h_t` into `c_t^KV` during training.
*   **Analogy:** Think of `h_t` as a high-resolution image and `c_t^KV` as a highly compressed JPEG. The JPEG is much smaller to store but retains the most important visual information. `c_t^KV` is the only part related to the token's *content* that gets stored in the cache.

---

#### Formulas (2), (3), and (4): Reconstructing the Final Key

The final Key for each attention head is constructed from two separate pieces: a "content" part and a "positional" part.

*   **Formula (2): Decompressing the "Content" Key**
    `[k_t,1^C; ...; k_t,nh^C] = W^UK * c_t^KV`
    *   This takes the small latent vector `c_t^KV` and projects it *back up* to the full dimension, creating the "content" part of the key (`k_t^C`) for all `n_h` attention heads.
    *   **`W^UK`:** This is a learned "Up-projection" matrix for Keys. It's the decompressor.

*   **Formula (3): Creating the "Positional" Key**
    `k_t^R = RoPE(W^KR * h_t)`
    *   This part handles the token's position in the sequence. It takes the *original* high-dimensional input `h_t` and applies a transformation (`W^KR`) followed by **Rotary Positional Embedding (RoPE)**.
    *   This creates a "decoupled" key `k_t^R` that purely encodes positional information. This is the second and final piece that gets stored in the cache.

*   **Formula (4): Combining for the Final Key**
    `k_t,i = [k_t,i^C; k_t^R]`
    *   The final key for a specific attention head `i` (`k_t,i`) is formed by simply concatenating (sticking together) the content part (`k_t,i^C`) and the positional part (`k_t^R`).

---

#### Formula (5): Decompressing the Value
`[v_t,1^C; ...; v_t,nh^C] = W^UV * c_t^KV`

*   This is very similar to the key decompression. It uses the *same* small latent vector `c_t^KV` but a *different* up-projection matrix (`W^UV`) to reconstruct the full-size Value vectors for all `n_h` heads.
*   This shows that `c_t^KV` is a **joint** compression of both Key and Value information.

**Key Takeaway for KV Cache:**
The text explicitly states that **only the blue-boxed vectors (`c_t^KV` and `k_t^R`) need to be cached.** This is the magic of MLA. Instead of storing massive Key and Value vectors for every head, you only store one tiny latent vector (`c_t^KV`) and one positional vector (`k_t^R`). The full Keys and Values are reconstructed on the fly when needed.

---

### Step 2: Processing Queries (Formulas 6-9)

This process mirrors the key generation, but it's for the Queries of the *current* token that will attend to the past keys in the cache.

*   **Formula (6): Compressing the Query**
    `c_t^Q = W^DQ * h_t`
    *   Just like for the KV, the input `h_t` is compressed into a small latent query vector `c_t^Q` using a separate down-projection matrix (`W^DQ`).

*   **Formula (7): Decompressing the "Content" Query**
    `[q_t,1^C; ...; q_t,nh^C] = W^UQ * c_t^Q`
    *   The small latent query `c_t^Q` is projected back up to create the "content" part of the query (`q_t^C`) for each head.

*   **Formula (8): Creating the "Positional" Query**
    `[q_t,1^R; ...; q_t,nh^R] = RoPE(W^QR * c_t^Q)`
    *   The positional part of the query (`q_t^R`) is created by applying RoPE to a projection of the *compressed* latent query `c_t^Q`.

*   **Formula (9): Combining for the Final Query**
    `q_t,i = [q_t,i^C; q_t,i^R]`
    *   The final query for each head `i` is formed by concatenating its content and positional parts.

### Summary of the Entire MLA Flow

1.  **For each token `t`:** Take its input embedding `h_t`.
2.  **Compress:** Create a tiny latent vector `c_t^KV` that jointly represents Keys and Values.
3.  **Get Position:** Create a positional key `k_t^R` from `h_t`.
4.  **Cache:** Store **only** `c_t^KV` and `k_t^R` in the KV cache. This is the **memory saving** step.
5.  **Attend:** When a new token needs to perform attention, it generates its query (`q_t,i`). It then retrieves the cached `c_s^KV` and `k_s^R` for all previous tokens `s`, reconstructs their full Keys and Values on the fly using the up-projection matrices, and computes the attention scores.

### How MLA Integrates with DeepSeek Sparse Attention

The beauty of this architecture is how MLA works seamlessly with DSA:

1. **DSA selects the relevant tokens:** The Lightning Indexer identifies the top-k most important previous tokens
2. **MLA processes only the selected tokens:** Instead of reconstructing Keys and Values for all 128,000 previous tokens, MLA only needs to decompress the cached `c_s^KV` and `k_s^R` for the selected top-k tokens
3. **Memory efficiency is multiplied:** DSA reduces the number of tokens to process, while MLA reduces the memory footprint of each token

This combination allows DeepSeek-V3.2 to process extremely long sequences (128,000+ tokens) while maintaining both computational efficiency and memory efficiency.
---

## Experimental Research Results

*Preliminary findings from [Open Superintelligence Lab](https://opensuperintelligencelab.com/) research*

### Research Questions

Our experiments aimed to answer:

1. **Does sparse attention improve performance on standard attention architectures?**
2. **Does sparse attention provide additional benefits when applied to already-efficient Multi-Head Latent Attention (MHLA)?**
3. **How do these mechanisms scale across different sequence lengths?**

Future research (that you can participate in):
## Core Architecture
1. **Why do we need extra weight for indexer score?** (`w_t,j^I` necessity)
2. **What is the optimal k value for different sequence lengths?**

## Lightning Indexer
3. **How does indexer performance scale with sequence length?**
4. **How does scaling influence indexer accuracy and computational efficiency?**


### Experiment 1: Standard Attention vs Sparse Attention

| Seq Length | Standard Loss | Sparse Loss | Improvement | Standard Acc | Sparse Acc |
|------------|---------------|-------------|-------------|--------------|------------|
| 64         | 8.52          | **3.56**    | **139% better** | 4.3%        | **53.2%**  |
| 128        | 7.28          | **3.00**    | **143% better** | 6.5%        | **57.6%**  |
| 256        | 7.15          | **1.78**    | **302% better** | 7.6%        | **68.4%**  |

**Key Finding**: Sparse attention dramatically outperformed standard attention, with benefits increasing for longer sequences.

### Experiment 2: MHLA Dense vs MHLA + Sparse

| Seq Length | MHLA Loss | MHLA+Sparse Loss | Improvement | MHLA Acc | MHLA+Sparse Acc |
|------------|-----------|------------------|-------------|----------|-----------------|
| 64         | 7.43      | **6.64**         | **12% better** | 9.2%     | **15.5%**       |
| 128        | 6.85      | 6.97             | -2% worse    | 10.3%    | 10.3%           |
| 256        | 6.61      | **6.55**         | **1% better** | 12.5%    | **13.2%**       |
| 1024       | **4.10**  | 6.91             | **-41% worse** | **32.2%** | 10.7%           |
| 2048       | 6.64      | **6.63**         | **0% same**   | 11.9%    | **14.4%**       |

**Key Finding**: Mixed results - sparse helped short sequences but significantly hurt long sequences on MHLA.

### Speed Analysis

**Experiment 1**: Similar training speeds (~0.06s per step for both)  
**Experiment 2**: Sparse version was 1-4% slower due to Lightning Indexer overhead

### Research Insights

**Why Sparse Helps Standard Attention:**
- **Forced selectivity** acts as regularization
- **Reduces attention dilution** in dense attention
- **Prevents overfitting** by focusing on relevant tokens

**Why Sparse May Not Help MHLA:**
- **Redundant mechanisms**: MHLA already compresses via latent space
- **Conflicting patterns**: MHLA's learned compression vs Lightning Indexer selection
- **Double compression**: May be too aggressive for long sequences

### Limitations and Caveats

These are preliminary results from limited experiments. Several factors may affect generalizability:

- **Limited training time**: Only 500-1000 steps per experiment
- **Small model size**: 512d models may not reflect larger model behavior
- **Dataset**: Results on TinyStories may not generalize to other domains
- **Hyperparameters**: Not extensively tuned for each configuration

### Conclusion

Our preliminary findings suggest:

1. **Sparse attention significantly improves standard attention architectures**
2. **MHLA's latent compression may already provide most benefits of sparsity**
3. **Combining both mechanisms may be redundant or even harmful for long sequences**

However, these results require further validation with larger models, longer training, and diverse datasets.

### About Open Superintelligence Lab

[Open Superintelligence Lab](https://opensuperintelligencelab.com/) is dedicated to advancing open-source AI research. We conduct experiments like these to understand fundamental mechanisms in large language models and share our findings transparently with the community.

Our research is ongoing, and we welcome collaboration and feedback from the community. These experiments represent active research that may contain flaws or limitations, and we encourage independent verification of our findings.

---

*This research is part of our ongoing investigation into efficient attention mechanisms. Results are preliminary and subject to revision as we conduct more extensive experiments.*
