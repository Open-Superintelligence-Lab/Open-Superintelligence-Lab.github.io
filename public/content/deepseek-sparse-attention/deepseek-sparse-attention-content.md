---
hero:
  title: "DeepSeek Sparse Attention"
  subtitle: "⚡ From quadratic to near-linear attention - The Lightning Indexer Breakthrough"
  tags:
    - "⏱️ Technical Deep Dive"
    - "📄 Research Article"
---

[Research Paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) • [Model](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) • [GitHub](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) • [Join open research on DeepSeek Sparse Attention](https://github.com/Open-Superintelligence-Lab/deepseek-sparse-attention-research)

> **🚦 Prerequisite:**  
> To get the most out of this article, you should have a basic understanding of **attention mechanism**.  
>  
> _Not sure what that means? Scroll down to the **Recommended Video Resource** below and get up to speed!_

The new DeepSeek architecture makes LLMs that went BRR now go BRRRRRRRRRRRR - it reduces attention complexity from quadratic scaling $O(L^2)$ to near-linear scaling $O(L \cdot k)$, where $L$ is the sequence length (number of tokens in the context window) and $k$ is a small constant (e.g., 2048).

![Inference Cost Comparison](/content/deepseek-sparse-attention/Inference-cost.jpeg)

DeepSeek-V3.2-Exp makes inference and training significantly cheaper through sparse attention, reducing computational costs while maintaining performance.

> ⚡ **Heads Up for Developers:**  
> This is probably an early signal for those building infrastructure to **get ready for DeepSeek-V4** which will likely use the same attention mechanism.

**📺 Recommended Video Resource:** For a comprehensive understanding of attention mechanisms and DeepSeek's Multihead Latent Attention, watch my course: [DeeepSeek V3 From Scratch](https://youtu.be/TfEG0TwueTs)

-  **If you're new to attention mechanisms:** Start from the beginning of the video until 58:39 where coding starts. Watching the coding part is optional.
-  **If you understand classic attention and want to only watch DeepSeek's Multihead Latent Attention (MLA):** Start from 38:53 or use this direct link: [https://youtu.be/TfEG0TwueTs?t=2333](https://youtu.be/TfEG0TwueTs?t=2333)
-  **Note:** I will explain MLA again in this article / video, but I recommend watching both for better understanding.

Standard Transformers use an "attention" mechanism where every new token being generated looks back at all the previous tokens in the sequence.

This is computationally very expensive. If you have a sequence of length $L$, the complexity is $O(L^2)$, meaning the computation and memory required grow quadratically.

Doubling the text length from 10,000 to 20,000 tokens doesn't just double the cost—it quadruples it. This makes processing very long documents (like books or large codebases) prohibitively slow and expensive.

Instead of having each token attend to all previous tokens, DeepSeek Sparse Attention (DSA) intelligently selects a small, fixed-size subset ($k$) of the most relevant previous tokens to attend to. This changes the complexity from $O(L^2)$ to $O(L \cdot k)$, which is much more manageable since $k$ is a small constant (e.g., 2048) and $L$ can be very large (e.g., 128,000 or 2,000,000).

![Attention Architecture](/content/deepseek-sparse-attention/Attention-architecture.png)

*Let's explain how DSA (marked in green) works with MLA (Multi-Head Latent Attention).*

DSA is made of two main components:

The **lightning indexer** is essentially a classic attention mechanism but much smaller and faster. It performs full attention between every token using ReLU activation (which is very fast) and much smaller dimensions for keys and queries.

#### Component 1: The Lightning Indexer

This is a fast and lightweight mechanism whose only job is to figure out which past tokens are important for the current token. It works exactly like standard attention: it calculates attention scores between tokens by performing dot products between queries (q) and keys (k) of those tokens. However, unlike regular attention where these scores are used to blend information/context from other tokens, the lightning indexer uses these scores purely to determine importance relationships between tokens. Score = Importance. The AI learns to generate keys and queries that calculate these importance scores accurately.

-  **How it works:** For the current token ($h_t$, where $t$ represents the current token index), the indexer quickly calculates an "index score" ($I_{t,s}$, where $t$ is the current token and $s$ is a previous token) for every previous token ($h_s$). This score represents the predicted relevance of token $s$ to token $t$.
-  **Formula (1):** The formula 1 is essentially a simplified attention calculation. It uses its own small set of queries ($q^I$) and keys ($k^I$) to compute these scores. ($I$) indicates that this belongs to indexer calculations.
-  **Why it's "Lightning":** It's designed for speed. It uses a simple $\text{ReLU}$ activation function and can be run with low-precision numbers (FP8), making it computationally very cheap, even though it still technically looks at all previous tokens (an $O(L^2)$ operation, but a very, very fast one).

### 1. The Formulas Explained (The "What")

The paper provides two key formulas that describe this two-step process.

#### **Formula (1): The Lightning Indexer**

$$
I_{t,s} = \sum_{j=1}^{H_I} w_{t,j}^I \cdot \text{ReLU}(q_{t,j}^I \cdot k_s^I)
$$

*Note: $j$ represents the attention head index in the Lightning Indexer, ranging from 1 to $H_I$ total heads.*

This formula calculates the **index score** ($I_{t,s}$), which represents the "relevance" of a past token $s$ to the current token $t$. Let's break it down:

*   $I_{t,s}$: The final importance score. A higher score means token $s$ is more important for token $t$.
*   $h_t$ and $h_s$: These are the vector representations (hidden states) of the current token ($t$) and a previous token ($s$).
*   $q_{t,j}^I$ and $k_s^I$: These are special, lightweight **query** and **key** vectors created just for the indexer (indicated by the $I$ superscript). They are derived from $h_t$ and $h_s$ respectively.
*   $q_{t,j}^I \cdot k_s^I$: This is a dot product, the fundamental operation in attention. It measures the similarity or compatibility between the query and the key.
*   $\text{ReLU}(\cdots)$: A simple activation function (Rectified Linear Unit). It's very fast to compute. If the dot product is negative, it becomes 0; otherwise, it stays the same.
*   $w_{t,j}^I$: An additional weight, also derived from the query token $h_t$. It acts as a learned gate or importance factor for each indexer head $j$.
*   $\sum \cdots$: This sums the results across all the indexer's heads ($H^I$). The indexer has only a few heads to keep it fast.

**In simple terms:** The Lightning Indexer is a mini, simplified attention mechanism. Its only job is to quickly calculate a relevance score for every pair of tokens without doing the full, expensive attention computation.

#### **Formula (2): The Main Attention Calculation**

$$
u_t = \text{Attn}(h_t, \{c_s | I_{t,s} \in \text{Top-k}(I_{t,:})\})
$$

This formula shows how the final output ($u_t$) is calculated after selecting the most relevant tokens. ($u_t$) is the same output you get from a standard attention - vector embedding of the current token enhanced with context (information) from previous tokens. 

[Attention Explained](https://youtu.be/wcDV3l4CD14?t=5562)

Like standard attention, $u_t$ combines the current token with contextual information from previous tokens—but instead of attending to all prior tokens, it only incorporates context from the Top-k most important ones as determined by the Lightning Indexer.

*   $u_t$: The final output hidden state for the current token $t$.
*   $\text{Attn}(\cdots)$: This represents the main, powerful attention mechanism (in this case, Multi-Query Attention).
*   $h_t$: (Somewhat confusingly) The query from the current token. In previous formula it was the current token itself, that is its vector representations (hidden state), now it's the query of the current token.
*   $\{c_s | I_{t,s} \in \text{Top-k}(I_{t,:})\}$: This is the most important part. It means: "Use the set of key-value entries $c_s$ **only if** their corresponding index score $I_{t,s}$ (calculated in Formula 1) is among the $\text{top-k}$ highest scores for the current token $t$."

**In simple terms:** The main attention mechanism is told to ignore almost all previous tokens and focus *only* on the handful of key-value entries that the Lightning Indexer identified as most important.

#### Component 2: The Fine-grained Token Selection
This component is simple: it takes all the index scores calculated by the Lightning Indexer and picks the $\text{top-k}$ highest scores.

-  **Function:** It acts as a gatekeeper. It tells the main, powerful attention mechanism: "You don't need to look at all 100,000 previous tokens. I've found the 2,048 most important ones for you. Just look at these."

The final attention output ($u_t$) is then calculated by the main attention module, but only using the current token's query and the $top-k$ key-value pairs that were selected.

### Step 3: How The Model Was Trained

The creation of DeepSeek-V3.2-Exp was not a matter of starting from scratch. Instead, researchers cleverly adapted an existing, powerful model, **DeepSeek-V3.1-Terminus**, which was already proficient in handling long contexts of 128K tokens. This adaptation involved a multi-stage training process designed to seamlessly integrate the new sparse attention mechanism while ensuring a fair comparison by using the same data distribution as the original model.

#### Phase 1: Continued Pre-Training

The first phase focused on teaching the model to use its new sparse attention architecture.

**Dense Warm-up Stage: An Initial Crash Course**

> **Goal:** To teach the brand-new **Lightning Indexer** what "important" tokens look like by having it mimic the original model's attention.

This was a short but critical initial stage lasting just 1,000 steps (2.1B tokens). The researchers froze the main model and kept the standard (dense) attention active. They then trained *only* the Lightning Indexer, tasking it with predicting the attention patterns of the powerful, pre-trained main model.
-   **Method:** A **KL-divergence loss** was used to measure how closely the indexer's predictions matched the main model's attention scores.
-   **Key Stats:**
    -   **Learning Rate:** $10^{-3}$
    -   **Batch Size:** 16 sequences of 128K tokens.

**Sparse Training Stage: Adapting to a New Reality**

> **Goal:** To adapt the entire model to work with the sparse attention pattern selected by the indexer.

This was the main training phase, lasting 15,000 steps (943.7B tokens). Here, the researchers "switched on" the sparse mechanism, un-froze the main model, and trained everything together.
-   **Method:** The model was now forced to predict the next word using only the **top-k** ($k=2048$) tokens identified by the indexer.
-   **Key Innovation:** The indexer and the main model were optimized separately by detaching the indexer from the main computational graph. This prevented their training signals from interfering.
    -   The **main model** was trained solely on language modeling loss (predicting the next word).
    -   The **Lightning Indexer** was trained solely on the KL-divergence loss to keep it aligned with the main model's attention, but only on the selected $k$ tokens.
-   **Key Stats:**
    -   **Learning Rate:** $7.3 \times 10^{-6}$
    -   **Batch Size:** 480 sequences of 128K tokens.

#### Phase 2: Post-Training - Fine-Tuning for a Fair Fight

To ensure a rigorous and fair assessment, the post-training pipeline—including algorithms and data—was kept identical to that of the original DeepSeek-V3.1-Terminus.

**Specialist Distillation**

First, the team developed specialized models for domains like mathematics, competitive programming, and agentic coding. Each specialist was fine-tuned from the same pre-trained DeepSeek-V3.2 base checkpoint. Using large-scale Reinforcement Learning (RL), these models generated high-quality, domain-specific data that was "distilled" to train the final model.

**Mixed RL Training**

Finally, the model was fine-tuned using **Group Relative Policy Optimization (GRPO)**. In a key strategic shift, the team merged reasoning, agent, and human alignment training into a single RL stage. This unified approach balanced performance across diverse skills while avoiding the "catastrophic forgetting" common in multi-stage training. The results were promising: the RL training curves of the new sparse model closely matched the original, demonstrating that DSA is a stable and effective addition.

---

*We are actively doing research on this ourselves - [contribute here](https://github.com/Open-Superintelligence-Lab/deepseek-sparse-attention-research)*

### Research Questions

Our experiments aimed to answer:

1. **Does sparse attention improve performance on standard attention architectures?**
2. **Does sparse attention provide additional benefits when applied to already-efficient Multi-Head Latent Attention (MHLA)?**
3. **How do these mechanisms scale across different sequence lengths?**

**Limited training time**: Only 500-1000 steps (5-10 minutes on 1 x Nvidia 4090) per experiment.

### Experiment 1: Standard Attention vs Sparse Attention

| Seq Length | Standard Loss | Sparse Loss | Improvement | Standard Acc | Sparse Acc |
|------------|---------------|-------------|-------------|--------------|------------|
| 64         | 8.52          | **3.56**    | **139% better** | 4.3%        | **53.2%**  |
| 128        | 7.28          | **3.00**    | **143% better** | 6.5%        | **57.6%**  |
| 256        | 7.15          | **1.78**    | **302% better** | 7.6%        | **68.4%**  |

**Key Finding**: Our small LLM learns faster with sparse attention (research in progress)

### Experiment 2: DeepSeek MHLA Dense vs MHLA + Sparse

| Seq Length | MHLA Loss | MHLA+Sparse Loss | Improvement | MHLA Acc | MHLA+Sparse Acc |
|------------|-----------|------------------|-------------|----------|-----------------|
| 64         | 7.43      | **6.64**         | **12% better** | 9.2%     | **15.5%**       |
| 128        | **6.85**  | 6.97             | -2% worse    | 10.3%    | 10.3%           |
| 256        | 6.61      | **6.55**         | **1% better** | 12.5%    | **13.2%**       |
| 1024       | **4.10**  | 6.91             | **-41% worse** | **32.2%** | 10.7%           |
| 2048       | 6.64      | **6.63**         | **0% same**   | 11.9%    | **14.4%**       |

**Key Finding**: Mixed results - sparse helped short sequences but hurt long sequences on MHLA. Might be due to implementation (research in progress)

### Speed Analysis

**Experiment 1**: Similar training speeds (~0.06s per step for both)  
**Experiment 2**: Sparse version was 1-4% slower due to Lightning Indexer overhead (shouldn't it be faster due to less tokens 🤔)

### Research Insights

Sparse attention maybe not just be a weaker dense attentin, it can show its own unique strangths, like preventing attention dilution.

### Future research (that you can participate in):

## Core Architecture
1. **Why do we need extra weight for indexer score?** ($w_{t,j}^I$ necessity)
2. **What is the optimal $k$ value for different sequence lengths?**

## Lightning Indexer
3. **How does indexer performance scale with sequence length?**
4. **How does scaling influence indexer accuracy and computational efficiency?**

### About Open Superintelligence Lab

[Open Superintelligence Lab](https://opensuperintelligencelab.com/) is dedicated to allowing anyone anywhere to contribute to open-source AI research. We conduct experiments like these to understand fundamental mechanisms neural networks and large language models and share our findings.

Our research is ongoing, and we welcome collaboration and feedback. These experiments represent active research that may contain flaws or limitations, and we encourage independent verification of our findings.

---

## Future Research Directions

![MHA and MQA Modes of MLA](/content/deepseek-sparse-attention/MHA-and-MQA-modes-of-MLA.png)

The diagram above illustrates Multi-Head Attention (MHA) and Multi-Query Attention (MQA) modes within the MLA framework.

---

*This research is part of our ongoing investigation into efficient attention mechanisms. Results are preliminary and subject to revision as we conduct more extensive experiments.*
