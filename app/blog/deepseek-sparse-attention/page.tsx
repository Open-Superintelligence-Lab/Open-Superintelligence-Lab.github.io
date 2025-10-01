'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";
import { MarkdownRenderer } from "@/components/markdown-renderer";

// Read the markdown content
const markdownContent = `# DeepSeek Sparse Attention

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

*   **How it works:** For the current token (\`h_t\`), the indexer quickly calculates an "index score" (\`I_t,s\`) for every previous token (\`h_s\`). This score represents the predicted relevance of token \`s\` to token \`t\`.
*   **Formula \`(1)\`:** The formula 1 is essentially a simplified attention calculation. It uses its own small set of queries (\`q_I\`) and keys (\`k_I\`) to compute these scores.
*   **Why it's "Lightning":** It's designed for speed. It uses a simple \`ReLU\` activation function and can be run with low-precision numbers (FP8), making it computationally very cheap, even though it still technically looks at all previous tokens (an \`O(L²)\` operation, but a very, very fast one).

### 1. The Formulas Explained (The "What")

The paper provides two key formulas that describe this two-step process.

#### **Formula (1): The Lightning Indexer**

\`I_t,s = Σ H_I (j=1) [w_t,j^I ⋅ ReLU(q_t,j^I ⋅ k_s^I)]\`

This formula calculates the **index score** (\`I_t,s\`), which represents the "relevance" of a past token \`s\` to the current token \`t\`. Let's break it down:

*   \`I_t,s\`: The final importance score. A higher score means token \`s\` is more important for token \`t\`.
*   \`h_t\` and \`h_s\`: These are the vector representations (hidden states) of the current token (\`t\`) and a previous token (\`s\`).
*   \`q_t,j^I\` and \`k_s^I\`: These are special, lightweight **query** and **key** vectors created just for the indexer (indicated by the \`I\` superscript). They are derived from \`h_t\` and \`h_s\` respectively.
*   \`q_t,j^I ⋅ k_s^I\`: This is a dot product, the fundamental operation in attention. It measures the similarity or compatibility between the query and the key.
*   \`ReLU(...)\`: A simple activation function (Rectified Linear Unit). It's very fast to compute. If the dot product is negative, it becomes 0; otherwise, it stays the same.
*   \`w_t,j^I\`: An additional weight, also derived from the query token \`h_t\`. It acts as a learned gate or importance factor for each indexer head \`j\`.
*   \`Σ ...\`: This sums the results across all the indexer's heads (\`H^I\`). The indexer has only a few heads to keep it fast.

**In simple terms:** The Lightning Indexer is a mini, simplified attention mechanism. Its only job is to quickly calculate a relevance score for every pair of tokens without doing the full, expensive attention computation.

#### **Formula (2): The Main Attention Calculation**

\`u_t = Attn(h_t, {c_s | I_t,s ∈ Top-k(I_t,:)})\`

This formula describes how the final output (\`u_t\`) is computed after the selection is done.

*   \`u_t\`: The final output hidden state for the current token \`t\`.
*   \`Attn(...)\`: This represents the main, powerful attention mechanism (in this case, Multi-Query Attention).
*   \`h_t\`: The query from the current token.
*   \`{c_s | I_t,s ∈ Top-k(I_t,:)}\`: This is the most important part. It means: "Use the set of key-value entries \`c_s\` **only if** their corresponding index score \`I_t,s\` (calculated in Formula 1) is among the \`top-k\` highest scores for the current token \`t\`."

**In simple terms:** The main attention mechanism is told to ignore almost all previous tokens and focus *only* on the handful of key-value entries that the Lightning Indexer identified as most important.

#### Component 2: The Fine-grained Token Selection
This component is simple: it takes all the index scores calculated by the Lightning Indexer and picks the \`top-k\` highest scores.

*   **Function:** It acts as a gatekeeper. It tells the main, powerful attention mechanism: "You don't need to look at all 100,000 previous tokens. I've found the 2,048 most important ones for you. Just look at these."

The final attention output (\`u_t\`) is then calculated by the main attention module, but only using the current token's query and the \`k\` key-value pairs that were selected.

### Step 3: How The Model Was Trained

They didn't train this model from scratch. They cleverly adapted an existing, powerful model (**DeepSeek-V3.1-Terminus**) that was already trained on long contexts. The training happened in several stages.

#### Stage 1: Continued Pre-Training (Two Phases)

1.  **Dense Warm-up Stage:**
    *   **Goal:** To teach the brand-new Lightning Indexer what "important" tokens look like.
    *   **Method:** They froze the main model and kept the standard (dense) attention active. They then trained *only* the Lightning Indexer. The indexer's objective was to make its importance scores match the attention scores from the powerful, pre-trained main model. They used a KL-divergence loss, which is a way of measuring how similar two probability distributions are. In essence, they told the indexer: "Learn to predict what the main model *would have* paid attention to." This phase was very short (1,000 steps).

2.  **Sparse Training Stage:**
    *   **Goal:** To adapt the entire model to work with the sparse attention pattern.
    *   **Method:** They "switched on" the \`top-k\` selector, making the attention sparse. They unfroze the main model and trained everything together.
        *   The **main model** was trained on its usual task: predicting the next word (language modeling loss). It had to learn to perform well with only the limited context provided by the selector.
        *   The **Lightning Indexer** continued to be trained with the KL-divergence loss to align with the main model's attention, but now only on the selected \`k\` tokens.
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

This section explains how the model takes the input for the current token (\`h_t\`) and creates the Key and Value vectors that will be stored (in a compressed form) and used by future tokens.

#### Formula (1): The Compression Step
\`c_t^KV = W^DKV * h_t\`

*   **What it does:** This is the most critical step for saving memory. It takes the large, high-dimensional input vector for the current token (\`h_t\`) and projects it down into a much smaller, low-dimensional vector called the **compressed latent vector** (\`c_t^KV\`).
*   **\`W^DKV\`:** This is a learned "Down-projection" matrix. The model learns how to best squish the information from \`h_t\` into \`c_t^KV\` during training.
*   **Analogy:** Think of \`h_t\` as a high-resolution image and \`c_t^KV\` as a highly compressed JPEG. The JPEG is much smaller to store but retains the most important visual information. \`c_t^KV\` is the only part related to the token's *content* that gets stored in the cache.

---

#### Formulas (2), (3), and (4): Reconstructing the Final Key

The final Key for each attention head is constructed from two separate pieces: a "content" part and a "positional" part.

*   **Formula (2): Decompressing the "Content" Key**
    \`[k_t,1^C; ...; k_t,nh^C] = W^UK * c_t^KV\`
    *   This takes the small latent vector \`c_t^KV\` and projects it *back up* to the full dimension, creating the "content" part of the key (\`k_t^C\`) for all \`n_h\` attention heads.
    *   **\`W^UK\`:** This is a learned "Up-projection" matrix for Keys. It's the decompressor.

*   **Formula (3): Creating the "Positional" Key**
    \`k_t^R = RoPE(W^KR * h_t)\`
    *   This part handles the token's position in the sequence. It takes the *original* high-dimensional input \`h_t\` and applies a transformation (\`W^KR\`) followed by **Rotary Positional Embedding (RoPE)**.
    *   This creates a "decoupled" key \`k_t^R\` that purely encodes positional information. This is the second and final piece that gets stored in the cache.

*   **Formula (4): Combining for the Final Key**
    \`k_t,i = [k_t,i^C; k_t^R]\`
    *   The final key for a specific attention head \`i\` (\`k_t,i\`) is formed by simply concatenating (sticking together) the content part (\`k_t,i^C\`) and the positional part (\`k_t^R\`).

---

#### Formula (5): Decompressing the Value
\`[v_t,1^C; ...; v_t,nh^C] = W^UV * c_t^KV\`

*   This is very similar to the key decompression. It uses the *same* small latent vector \`c_t^KV\` but a *different* up-projection matrix (\`W^UV\`) to reconstruct the full-size Value vectors for all \`n_h\` heads.
*   This shows that \`c_t^KV\` is a **joint** compression of both Key and Value information.

**Key Takeaway for KV Cache:**
The text explicitly states that **only the blue-boxed vectors (\`c_t^KV\` and \`k_t^R\`) need to be cached.** This is the magic of MLA. Instead of storing massive Key and Value vectors for every head, you only store one tiny latent vector (\`c_t^KV\`) and one positional vector (\`k_t^R\`). The full Keys and Values are reconstructed on the fly when needed.

---

### Step 2: Processing Queries (Formulas 6-9)

This process mirrors the key generation, but it's for the Queries of the *current* token that will attend to the past keys in the cache.

*   **Formula (6): Compressing the Query**
    \`c_t^Q = W^DQ * h_t\`
    *   Just like for the KV, the input \`h_t\` is compressed into a small latent query vector \`c_t^Q\` using a separate down-projection matrix (\`W^DQ\`).

*   **Formula (7): Decompressing the "Content" Query**
    \`[q_t,1^C; ...; q_t,nh^C] = W^UQ * c_t^Q\`
    *   The small latent query \`c_t^Q\` is projected back up to create the "content" part of the query (\`q_t^C\`) for each head.

*   **Formula (8): Creating the "Positional" Query**
    \`[q_t,1^R; ...; q_t,nh^R] = RoPE(W^QR * c_t^Q)\`
    *   The positional part of the query (\`q_t^R\`) is created by applying RoPE to a projection of the *compressed* latent query \`c_t^Q\`.

*   **Formula (9): Combining for the Final Query**
    \`q_t,i = [q_t,i^C; q_t,i^R]\`
    *   The final query for each head \`i\` is formed by concatenating its content and positional parts.

### Summary of the Entire MLA Flow

1.  **For each token \`t\`:** Take its input embedding \`h_t\`.
2.  **Compress:** Create a tiny latent vector \`c_t^KV\` that jointly represents Keys and Values.
3.  **Get Position:** Create a positional key \`k_t^R\` from \`h_t\`.
4.  **Cache:** Store **only** \`c_t^KV\` and \`k_t^R\` in the KV cache. This is the **memory saving** step.
5.  **Attend:** When a new token needs to perform attention, it generates its query (\`q_t,i\`). It then retrieves the cached \`c_s^KV\` and \`k_s^R\` for all previous tokens \`s\`, reconstructs their full Keys and Values on the fly using the up-projection matrices, and computes the attention scores.

### How MLA Integrates with DeepSeek Sparse Attention

The beauty of this architecture is how MLA works seamlessly with DSA:

1. **DSA selects the relevant tokens:** The Lightning Indexer identifies the top-k most important previous tokens
2. **MLA processes only the selected tokens:** Instead of reconstructing Keys and Values for all 128,000 previous tokens, MLA only needs to decompress the cached \`c_s^KV\` and \`k_s^R\` for the selected top-k tokens
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
1. **Why do we need extra weight for indexer score?** (\`w_t,j^I\` necessity)
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

*This research is part of our ongoing investigation into efficient attention mechanisms. Results are preliminary and subject to revision as we conduct more extensive experiments.*`;

export default function DeepSeekProject() {
  const { language } = useLanguage();

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Background effects */}
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
        </div>
        
        {/* Animated background particles */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/6 left-1/6 w-3 h-3 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full opacity-60 animate-pulse"></div>
          <div className="absolute top-1/4 right-1/5 w-2 h-2 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full opacity-50 animate-pulse delay-300"></div>
          <div className="absolute top-1/3 left-1/8 w-4 h-4 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full opacity-40 animate-pulse delay-700"></div>
          <div className="absolute bottom-1/4 right-1/6 w-2.5 h-2.5 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full opacity-55 animate-pulse delay-1000"></div>
        </div>
        
        <div className="relative container mx-auto px-6 pt-32 pb-24">
          <div className="text-center max-w-4xl mx-auto">
            <div className="relative">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium mb-8 leading-tight">
                <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                  {language === 'en' ? 'DeepSeek\'s Attention Revolution' : 'DeepSeek 的注意力革命'}
                </span>
              </h1>
              <div className="text-lg md:text-xl text-slate-400 mb-4">
                {language === 'en' 
                  ? '⚡ From O(L²) to O(Lk) - The Lightning Indexer Breakthrough'
                  : '⚡ 从 O(L²) 到 O(Lk) - 闪电索引器突破'
                }
              </div>
              
              {/* Glow effect for the title */}
              <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-medium leading-tight blur-sm">
                <span className="bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-cyan-400/20 bg-clip-text text-transparent">
                  {language === 'en' ? 'DeepSeek\'s Attention Revolution' : 'DeepSeek 的注意力革命'}
                </span>
              </div>
            </div>
            
            <p className="text-xl text-slate-300 mb-12 leading-relaxed">
              {language === 'en' 
                ? 'A deep dive into sparse attention and the Lightning Indexer - DeepSeek-V3.2-Exp'
                : '深入探讨稀疏注意力和闪电索引器 - DeepSeek-V3.2-Exp'
              }
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 min-h-screen py-20">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          {/* Article Container */}
          <article className="max-w-4xl mx-auto">
            {/* Content Card */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl shadow-2xl overflow-hidden">
              {/* Article Header */}
              <div className="bg-gradient-to-r from-blue-600/10 via-purple-600/10 to-cyan-600/10 border-b border-white/10 px-8 sm:px-12 py-8">
                <div className="flex items-center gap-3 text-sm text-slate-400 mb-4">
                  <span className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Technical Deep Dive
                  </span>
                  <span className="text-slate-600">•</span>
                  <span className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Research Article
                  </span>
                </div>
                <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent mb-4">
                  DeepSeek Sparse Attention
                </h1>
                <p className="text-slate-400 text-lg leading-relaxed">
                  A comprehensive exploration of the Lightning Indexer and Multi-Head Latent Attention architecture
                </p>
              </div>

              {/* Article Body */}
              <div className="px-8 sm:px-12 py-12">
                <div className="prose prose-lg prose-invert max-w-none">
            <MarkdownRenderer content={markdownContent} />
                </div>
              </div>

              {/* Article Footer */}
              <div className="bg-gradient-to-r from-blue-600/5 via-purple-600/5 to-cyan-600/5 border-t border-white/10 px-8 sm:px-12 py-8">
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                  <div className="flex items-center gap-3 text-sm text-slate-400">
                    <span className="flex items-center gap-2">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                      Open Superintelligence Lab
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Share</span>
                    <a href="https://twitter.com/intent/tweet?text=DeepSeek%20Sparse%20Attention%20-%20Lightning%20Indexer" 
                       target="_blank" 
                       rel="noopener noreferrer"
                       className="text-slate-400 hover:text-blue-400 transition-colors">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                      </svg>
                    </a>
                    <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://opensuperintelligencelab.com" 
                       target="_blank" 
                       rel="noopener noreferrer"
                       className="text-slate-400 hover:text-blue-400 transition-colors">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                      </svg>
                    </a>
                  </div>
                </div>
              </div>
          </div>

            {/* Navigation */}
            <div className="mt-12 flex flex-col sm:flex-row items-center justify-between gap-4">
            <Link 
              href="/"
                className="group flex items-center gap-2 px-6 py-3 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-500/50 text-slate-300 hover:text-blue-400 font-medium rounded-xl transition-all duration-300"
            >
                <svg className="w-5 h-5 group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              {language === 'en' ? 'Back to Home' : '返回首页'}
            </Link>
              
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <span className="hidden sm:inline">Scroll to</span>
                <button 
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                  className="flex items-center gap-1 px-4 py-2 hover:text-blue-400 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                  </svg>
                  Top
                </button>
              </div>
          </div>
          </article>
        </div>
      </main>
    </>
  );
}
