'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";
import { MarkdownRenderer } from "@/components/markdown-renderer";

// Read the markdown content
const markdownContent = `# DeepSeek's Attention Revolution: From O(L²) to O(Lk)

## Key Innovation: DeepSeek Sparse Attention (DSA)

**Problem:** Standard attention scales quadratically O(L²) with sequence length, making long contexts extremely expensive.

**Solution:** DSA reduces complexity to nearly linear O(Lk) by intelligently selecting only the most relevant tokens.

## How It Works

### 1. Lightning Indexer
- Fast "scout" that identifies the most important previous tokens
- Uses minimal computation to score token relevance
- Runs in low-precision FP8 format for efficiency

### 2. Top-k Selection
- Selects only the top 2048 most relevant tokens
- Main attention operates only on these selected tokens
- Dramatically reduces computational cost

## Training Process

1. **Dense Warm-up:** Train indexer to mimic full attention patterns
2. **Sparse Training:** Adapt entire model to work with sparse attention
3. **Post-Training:** Fine-tune for specific tasks using same methods as dense model

## Results

- **Performance:** Nearly identical to dense model on benchmarks
- **Efficiency:** Massive cost reduction for long-context processing
- **Scalability:** Linear cost growth instead of quadratic

## Technical Details

### The Tyranny of O(L²)

Standard attention requires every token to compute a relationship with ALL previous tokens in the sequence. This creates an L×L attention matrix, leading to O(L²) complexity.

**Examples:**
- 1,000 tokens: 1,000,000 computations
- 10,000 tokens: 100,000,000 computations  
- 128,000 tokens: 16,384,000,000 computations

### The Lightning Indexer Solution

The Lightning Indexer uses a two-phase approach:

**Phase 1: The Scout**
- Lightning Indexer quickly scores token relevance
- Formula: \`I_t,s = Σ w_t,j^I * ReLU(q_t,j^I ⋅ k_s^I)\`
- Runs in FP8 precision for speed
- Minimal computational cost

**Phase 2: Elite Squad**
- Top-k selection chooses only the 2048 most relevant tokens
- Formula: \`u_t = Attn(h_t, {c_s | I_t,s ∈ Top-k})\`
- Full attention operates only on selected tokens
- Complexity: O(L²) → O(L×k)

### Complexity Transformation

This is the key breakthrough: reducing quadratic scaling to nearly linear scaling.

- **Traditional:** O(L²) - grows quadratically
- **Sparse:** O(Lk) - grows linearly (k is fixed at 2048)

**Real-world impact:**
- L=1,000: 1M → 2M operations
- L=10,000: 100M → 20M operations
- L=128,000: 16B → 262M operations

## Training Methodology

### Phase 1: Dense Warm-up
- Freeze main model, only indexer trains
- Dense attention provides ground truth
- KL divergence loss measures distribution difference
- Duration: 1,000 steps

### Phase 2: Sparse Training
- Enable Top-k selection (k=2048)
- Train both model and indexer together
- Language modeling + KL loss
- Duration: 15,000 steps
- Data: 943.7 billion tokens

## Performance Results

### Efficiency Gains
- **Prefilling:** 10-50x faster
- **Decoding:** 5-20x faster
- **Cost:** 90% reduction
- **Scaling:** Linear instead of quadratic

### Quality Preservation
- **Math:** 99.1% of dense performance
- **Coding:** 99.3% of dense performance
- **General:** 99.0% of dense performance

## Implications

### Democratized AI
- Long-context AI becomes economically viable
- Real-time processing of entire books
- Accessible to everyone, not just large corporations

### Research Impact
- Opens new possibilities for long-context applications
- Enables processing of entire codebases
- Makes document analysis and summarization practical

### Future Directions
- Further optimization of the indexer
- Dynamic k selection based on context
- Integration with other efficiency techniques

## Resources

- **Paper:** [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)
- **Code:** [DeepSeek Research Repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- **Community:** Join our research community to explore and contribute to sparse attention research

## Conclusion

DeepSeek's Sparse Attention represents a fundamental breakthrough in transformer efficiency. By reducing complexity from O(L²) to O(Lk), it makes long-context AI economically viable while maintaining nearly identical performance. This opens new possibilities for applications that require processing entire documents, codebases, or conversations without the prohibitive costs of traditional attention mechanisms.

The Lightning Indexer approach demonstrates that intelligent token selection can dramatically improve efficiency without sacrificing quality, paving the way for a new generation of long-context AI applications.`;

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
      <main className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
        <div className="container mx-auto px-6 max-w-4xl">
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <MarkdownRenderer content={markdownContent} />
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
