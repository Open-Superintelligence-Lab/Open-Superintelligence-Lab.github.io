'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";
import { MarkdownRenderer } from "@/components/markdown-renderer";

const tutorialContent = `### High-Level Summary

The paper introduces an experimental model, **DeepSeek-V3.2-Exp**, which is a more efficient version of its predecessor, **DeepSeek-V3.1-Terminus**. The core problem with large language models is that the computational cost of the attention mechanism grows quadratically with the length of the input sequence (O(L²)), making very long contexts (like a whole book) extremely expensive to process.

The solution presented is **DeepSeek Sparse Attention (DSA)**, a new attention mechanism that reduces this complexity to be nearly linear (O(Lk), where k is a small constant). They achieve this without a significant drop in performance, making long-context processing much faster and cheaper.

---

### Step 1: The Problem - The Cost of Long Contexts

Standard self-attention, the core of the Transformer architecture, requires every token in a sequence to "attend to" (or compare itself with) every single token that came before it.

*   If you have a sequence of **L** tokens, the 10th token looks at the first 9 tokens.
*   The 100th token looks at the first 99 tokens.
*   The 100,000th token looks at the first 99,999 tokens.

This results in a total number of computations proportional to L², which becomes computationally infeasible and very expensive for long sequences (e.g., L = 128,000).

### Step 2: The Solution - DeepSeek Sparse Attention (DSA)

DSA is the key innovation. Instead of having every token look at *all* previous tokens, it intelligently selects only a small, fixed number (\\`k\\`) of the most relevant previous tokens to look at. This is a two-part process.

#### Part A: The "Lightning Indexer" (The Scout)

This is a very small, fast component whose only job is to quickly figure out which previous tokens are most important for the current token.

*   For a current query token (\\`h_t\\`), the indexer calculates an "index score" (\\`I_t,s\\`) for every preceding token (\\`h_s\\`).
*   This score represents the predicted relevance of token \\`s\\` to token \\`t\\`.
*   As described in **Equation (1)**, this calculation is designed to be extremely fast. It uses a small number of heads and can even run in low-precision FP8 format, making it much cheaper than full attention.
*   Think of it as a "scout" that quickly scans the entire history and flags the most promising locations.

#### Part B: Fine-grained Token Selection & Sparse Attention (The Main Operation)

Once the Lightning Indexer has calculated scores for all preceding tokens, this mechanism kicks in.

*   It simply picks the **top-k** highest scores. For this model, \\`k\\` is set to **2048**.
*   The main attention mechanism then operates *only* on the key-value pairs of these 2048 selected tokens.
*   Instead of calculating attention over L tokens, it now only calculates it over \\`k\\` tokens. This dramatically reduces the complexity from O(L²) to O(L * k). Since \\`k\\` is a fixed number and much smaller than \\`L\\`, the cost grows linearly with the sequence length, not quadratically.

**Figure 1** in the paper visualizes this. The input (\\`h_t\\`) is split. One path goes to the Lightning Indexer to get the scores. The other path goes to the main attention module. The indexer's output is used by a "Top-k Selector" to filter the key-value pairs that the main attention module is allowed to see.



### Step 3: The Training Process - How to Teach the Model to be Sparse

They couldn't just switch on DSA in a pre-trained model and expect it to work. They used a careful, multi-stage training process, starting from an already powerful model (DeepSeek-V3.1-Terminus).

#### Stage 1: Dense Warm-up (Teaching the Scout)

The first step was to train just the Lightning Indexer.
*   **Goal:** Teach the indexer to find the same tokens that the full, dense attention mechanism would find important.
*   **Method:** They froze the main model parameters and kept the standard (dense) attention active. They then trained the indexer to mimic the attention patterns of the main model.
*   **Loss Function (Equation 3):** They used a KL-divergence loss, which essentially measures how different two probability distributions are. The goal was to minimize the difference between the indexer's scores and the actual attention scores from the main model.
*   This stage was very short (1000 steps), just to get the indexer properly initialized.

#### Stage 2: Sparse Training (Adapting the Whole System)

Now, they activate the full DSA mechanism, including the top-k selection.
*   **Goal:** Adapt the entire model (both the main part and the indexer) to work effectively with this new sparse attention pattern.
*   **Method:** The model now only "sees" the 2048 tokens selected by the indexer.
    *   The **main model** is trained on the standard language modeling task (predicting the next token).
    *   The **Lightning Indexer** continues to be trained to align with the main attention distribution, but now only on the set of selected tokens (as shown in **Equation 4**).
*   This was the main training phase, running for 15,000 steps on a massive amount of data (943.7 billion tokens).

#### Stage 3: Post-Training (Fine-tuning and Alignment)

After the model learned to use sparse attention, they fine-tuned it for specific tasks like coding, math, and following instructions. Crucially, they used the **exact same data and methods** as they did for the non-sparse DeepSeek-V3.1-Terminus. This ensures a fair comparison of the models' capabilities, isolating the impact of adding DSA.

### Step 4: The Results - The Payoff

The paper evaluates the new model on two fronts: capabilities and efficiency.

#### Capabilities (Table 1 & Figure 2)

*   **Performance:** DeepSeek-V3.2-Exp performs **almost identically** to its dense predecessor, DeepSeek-V3.1-Terminus. There is no significant drop in quality on benchmarks for math, coding, and general knowledge.
*   **Training Stability:** The training curves in Figure 2 show that the sparse model learns just as steadily during Reinforcement Learning (RL) fine-tuning as the dense model. This proves that DSA is a stable architecture.

#### Efficiency (Figure 3)

This is the main victory. The graphs show the cost per million tokens during inference.
*   **Prefilling (Processing the prompt):** As the input context gets longer (moving right on the x-axis), the cost for the old model (blue line) skyrockets. The cost for the new sparse model (orange line) grows much, much slower.
*   **Decoding (Generating the response):** The same pattern holds. The cost of generating a new token is significantly lower with the sparse model when the context is long, as it doesn't need to re-scan the entire history with expensive, dense attention.

In summary, they successfully traded a tiny, almost negligible amount of model performance for a massive improvement in computational efficiency for long-context tasks.`;

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
                  {language === 'en' ? 'How DeepSeek Slashed LLM Costs' : 'DeepSeek 如何大幅降低 LLM 成本'}
                </span>
              </h1>
              
              {/* Glow effect for the title */}
              <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-medium leading-tight blur-sm">
                <span className="bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-cyan-400/20 bg-clip-text text-transparent">
                  {language === 'en' ? 'How DeepSeek Slashed LLM Costs' : 'DeepSeek 如何大幅降低 LLM 成本'}
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
      <main className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
        <div className="container mx-auto px-6 max-w-4xl">
          
          {/* Research Paper Link */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 mb-8">
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl flex items-center justify-center">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-bold text-white mb-4">
                  {language === 'en' ? 'Read the Original Research Paper' : '阅读原始研究论文'}
                </h2>
                <p className="text-slate-300 mb-6 leading-relaxed">
                  {language === 'en' 
                    ? 'Explore the full DeepSeek-V3.2-Exp research paper for complete technical details and experimental results.'
                    : '探索完整的 DeepSeek-V3.2-Exp 研究论文，获取完整的技术细节和实验结果。'
                  }
                </p>
                <a 
                  href="https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-blue-500/25"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  {language === 'en' ? 'Read Research Paper' : '阅读研究论文'}
                </a>
              </div>
            </div>
          </div>

          {/* Tutorial Content */}
          <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-8 mb-8">
            <MarkdownRenderer content={tutorialContent} />
          </div>

          {/* Back to Home */}
          <div className="text-center">
            <Link 
              href="/"
              className="inline-flex items-center gap-2 px-6 py-3 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-blue-500 hover:text-blue-400 transition-all duration-300 transform hover:scale-105"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              {language === 'en' ? 'Back to Home' : '返回首页'}
            </Link>
          </div>
        </div>
      </main>
    </>
  );
}
