'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { useState } from "react";

const tutorialContent = `## Key Innovation: DeepSeek Sparse Attention (DSA)

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
- **Scalability:** Linear cost growth instead of quadratic`;

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
          
          {/* The Problem - Visual Explanation */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">⚠️</span>
                {language === 'en' ? 'The Tyranny of O(L²)' : 'O(L²) 的暴政'}
              </h2>
              <p className="text-slate-400 text-lg">
                {language === 'en' 
                  ? 'Why long contexts break the bank'
                  : '为什么长上下文会掏空钱包'
                }
              </p>
            </div>
            
            <div className="grid md:grid-cols-3 gap-6 mb-8">
              {/* Sequence Length Examples */}
              <div className="bg-gradient-to-br from-red-900/20 to-red-800/20 backdrop-blur-sm border border-red-600/30 rounded-xl p-6 text-center">
                <div className="text-4xl mb-4">📄</div>
                <h3 className="text-xl font-bold text-white mb-2">1,000 tokens</h3>
                <div className="text-red-400 font-mono text-2xl mb-2">1M ops</div>
                <p className="text-slate-300 text-sm">Short document</p>
              </div>
              
              <div className="bg-gradient-to-br from-orange-900/20 to-orange-800/20 backdrop-blur-sm border border-orange-600/30 rounded-xl p-6 text-center">
                <div className="text-4xl mb-4">📚</div>
                <h3 className="text-xl font-bold text-white mb-2">10,000 tokens</h3>
                <div className="text-orange-400 font-mono text-2xl mb-2">100M ops</div>
                <p className="text-slate-300 text-sm">Medium document</p>
              </div>
              
              <div className="bg-gradient-to-br from-red-900/20 to-red-800/20 backdrop-blur-sm border border-red-600/30 rounded-xl p-6 text-center">
                <div className="text-4xl mb-4">🏢</div>
                <h3 className="text-xl font-bold text-white mb-2">128,000 tokens</h3>
                <div className="text-red-400 font-mono text-2xl mb-2">16B ops</div>
                <p className="text-slate-300 text-sm">Entire codebase</p>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6 text-center">
              <div className="flex items-center justify-center gap-4 mb-4">
                <span className="text-slate-300">Every token must look at</span>
                <div className="bg-red-500/20 border border-red-500/50 rounded-lg px-4 py-2">
                  <span className="text-red-400 font-mono text-lg">ALL</span>
                </div>
                <span className="text-slate-300">previous tokens</span>
              </div>
              <p className="text-slate-400 text-sm">
                {language === 'en' 
                  ? 'This quadratic scaling makes long contexts prohibitively expensive'
                  : '这种二次缩放使得长上下文极其昂贵'
                }
              </p>
            </div>
          </div>

          {/* The Solution - Lightning Indexer */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">⚡</span>
                {language === 'en' ? 'The Lightning Indexer Solution' : '闪电索引器解决方案'}
              </h2>
              <p className="text-slate-400 text-lg">
                {language === 'en' 
                  ? 'From O(L²) to O(Lk) - The scout and elite squad approach'
                  : '从 O(L²) 到 O(Lk) - 侦察兵和精英小队方法'
                }
              </p>
            </div>
            
            <div className="grid md:grid-cols-2 gap-8 mb-8">
              {/* Scout Phase */}
              <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 backdrop-blur-sm border border-blue-600/30 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
                    <span className="text-2xl">🔍</span>
                  </div>
                  <h3 className="text-xl font-bold text-white">
                    {language === 'en' ? 'Phase 1: The Scout' : '阶段 1：侦察兵'}
                  </h3>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="text-blue-400 font-mono text-sm">Lightning Indexer</span>
                    <span className="text-slate-400 text-sm">→</span>
                    <span className="text-slate-300 text-sm">
                      {language === 'en' ? 'Fast relevance scoring' : '快速相关性评分'}
                    </span>
                  </div>
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                    <div className="text-blue-400 font-mono text-sm mb-1">I_t,s = Σ w_t,j^I * ReLU(q_t,j^I ⋅ k_s^I)</div>
                    <p className="text-slate-300 text-xs">
                      {language === 'en' 
                        ? 'Ultra-fast dot product + ReLU for speed'
                        : '超快速点积 + ReLU 以提高速度'
                      }
                    </p>
                  </div>
                  <div className="flex items-center gap-2 text-slate-300 text-sm">
                    <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                    {language === 'en' ? 'Runs in FP8 precision' : '以 FP8 精度运行'}
                  </div>
                  <div className="flex items-center gap-2 text-slate-300 text-sm">
                    <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                    {language === 'en' ? 'Minimal computational cost' : '最小计算成本'}
                  </div>
                </div>
              </div>
              
              {/* Elite Squad Phase */}
              <div className="bg-gradient-to-br from-emerald-900/20 to-emerald-800/20 backdrop-blur-sm border border-emerald-600/30 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-lg flex items-center justify-center">
                    <span className="text-2xl">🎯</span>
                  </div>
                  <h3 className="text-xl font-bold text-white">
                    {language === 'en' ? 'Phase 2: Elite Squad' : '阶段 2：精英小队'}
                  </h3>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="text-emerald-400 font-mono text-sm">Top-k Selection</span>
                    <span className="text-slate-400 text-sm">→</span>
                    <span className="text-slate-300 text-sm">
                      {language === 'en' ? 'k=2048 best tokens' : 'k=2048 个最佳 token'}
                    </span>
                  </div>
                  <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-3">
                    <div className="text-emerald-400 font-mono text-sm mb-1">u_t = Attn(h_t, &#123;c_s | I_t,s ∈ Top-k&#125;)</div>
                    <p className="text-slate-300 text-xs">
                      {language === 'en' 
                        ? 'Full attention only on selected tokens'
                        : '仅对选定的 token 进行完整注意力'
                      }
                    </p>
                  </div>
                  <div className="flex items-center gap-2 text-slate-300 text-sm">
                    <span className="w-2 h-2 bg-emerald-400 rounded-full"></span>
                    {language === 'en' ? 'L×L → L×k complexity' : 'L×L → L×k 复杂度'}
                  </div>
                  <div className="flex items-center gap-2 text-slate-300 text-sm">
                    <span className="w-2 h-2 bg-emerald-400 rounded-full"></span>
                    {language === 'en' ? 'Massive cost reduction' : '大幅降低成本'}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Complexity Comparison */}
            <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6">
              <div className="text-center mb-4">
                <h3 className="text-xl font-bold text-white mb-2">
                  {language === 'en' ? 'Complexity Transformation' : '复杂度转换'}
                </h3>
              </div>
              <div className="flex items-center justify-center gap-6">
                <div className="text-center">
                  <div className="text-red-400 font-mono text-2xl mb-2">O(L²)</div>
                  <div className="text-slate-300 text-sm">
                    {language === 'en' ? 'Quadratic' : '二次'}
                  </div>
                </div>
                <div className="text-3xl text-slate-400">→</div>
                <div className="text-center">
                  <div className="text-emerald-400 font-mono text-2xl mb-2">O(Lk)</div>
                  <div className="text-slate-300 text-sm">
                    {language === 'en' ? 'Nearly Linear' : '近似线性'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Training Process */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">🎓</span>
                {language === 'en' ? 'Training the Sparse Model' : '训练稀疏模型'}
              </h2>
              <p className="text-slate-400 text-lg">
                {language === 'en' 
                  ? 'Two-phase approach: Teach the scout, then train the team'
                  : '两阶段方法：训练侦察兵，然后训练团队'
                }
              </p>
            </div>
            
            <div className="grid md:grid-cols-2 gap-6">
              {/* Phase 1 */}
              <div className="bg-gradient-to-br from-purple-900/20 to-purple-800/20 backdrop-blur-sm border border-purple-600/30 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold">
                    1
                  </div>
                  <h3 className="text-lg font-bold text-white">
                    {language === 'en' ? 'Dense Warm-up' : '密集预热'}
                  </h3>
                </div>
                <div className="space-y-3 text-sm">
                  <div className="flex items-center gap-2 text-slate-300">
                    <span className="w-2 h-2 bg-purple-400 rounded-full"></span>
                    {language === 'en' ? 'Freeze main model' : '冻结主模型'}
                  </div>
                  <div className="flex items-center gap-2 text-slate-300">
                    <span className="w-2 h-2 bg-purple-400 rounded-full"></span>
                    {language === 'en' ? 'Train indexer to mimic dense attention' : '训练索引器模仿密集注意力'}
                  </div>
                  <div className="flex items-center gap-2 text-slate-300">
                    <span className="w-2 h-2 bg-purple-400 rounded-full"></span>
                    {language === 'en' ? 'KL divergence loss' : 'KL 散度损失'}
                  </div>
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-2 mt-3">
                    <div className="text-purple-400 font-mono text-xs">1,000 steps</div>
                  </div>
                </div>
              </div>
              
              {/* Phase 2 */}
              <div className="bg-gradient-to-br from-teal-900/20 to-teal-800/20 backdrop-blur-sm border border-teal-600/30 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-teal-500 to-teal-600 rounded-lg flex items-center justify-center text-white font-bold">
                    2
                  </div>
                  <h3 className="text-lg font-bold text-white">
                    {language === 'en' ? 'Sparse Training' : '稀疏训练'}
                  </h3>
                </div>
                <div className="space-y-3 text-sm">
                  <div className="flex items-center gap-2 text-slate-300">
                    <span className="w-2 h-2 bg-teal-400 rounded-full"></span>
                    {language === 'en' ? 'Enable Top-k selection' : '启用 Top-k 选择'}
                  </div>
                  <div className="flex items-center gap-2 text-slate-300">
                    <span className="w-2 h-2 bg-teal-400 rounded-full"></span>
                    {language === 'en' ? 'Train both model and indexer' : '训练模型和索引器'}
                  </div>
                  <div className="flex items-center gap-2 text-slate-300">
                    <span className="w-2 h-2 bg-teal-400 rounded-full"></span>
                    {language === 'en' ? 'Language modeling + KL loss' : '语言建模 + KL 损失'}
                  </div>
                  <div className="bg-teal-500/10 border border-teal-500/30 rounded-lg p-2 mt-3">
                    <div className="text-teal-400 font-mono text-xs">15,000 steps</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Results Summary */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">🏆</span>
                {language === 'en' ? 'The Results' : '结果'}
              </h2>
              <p className="text-slate-400 text-lg">
                {language === 'en' 
                  ? 'Massive efficiency gains with minimal performance loss'
                  : '巨大的效率提升，性能损失最小'
                }
              </p>
            </div>
            
            <div className="grid md:grid-cols-2 gap-6">
              {/* Performance */}
              <div className="bg-gradient-to-br from-green-900/20 to-green-800/20 backdrop-blur-sm border border-green-600/30 rounded-xl p-6 text-center">
                <div className="text-4xl mb-4">📊</div>
                <h3 className="text-xl font-bold text-white mb-2">
                  {language === 'en' ? 'Performance' : '性能'}
                </h3>
                <div className="text-green-400 text-2xl font-bold mb-2">~99%</div>
                <p className="text-slate-300 text-sm">
                  {language === 'en' 
                    ? 'Identical to dense model on benchmarks'
                    : '在基准测试中与密集模型相同'
                  }
                </p>
              </div>
              
              {/* Efficiency */}
              <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 backdrop-blur-sm border border-blue-600/30 rounded-xl p-6 text-center">
                <div className="text-4xl mb-4">⚡</div>
                <h3 className="text-xl font-bold text-white mb-2">
                  {language === 'en' ? 'Efficiency' : '效率'}
                </h3>
                <div className="text-blue-400 text-2xl font-bold mb-2">~10x</div>
                <p className="text-slate-300 text-sm">
                  {language === 'en' 
                    ? 'Faster for long contexts'
                    : '长上下文更快'
                  }
                </p>
              </div>
            </div>
          </div>

          {/* Community Tasks */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 mb-8">
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0">
                <div className="w-16 h-16 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-bold text-white mb-4">
                  {language === 'en' ? 'Community Research Tasks' : '社区研究任务'}
                </h2>
                <p className="text-slate-300 mb-6 leading-relaxed">
                  {language === 'en' 
                    ? 'Join our research community to explore DeepSeek Sparse Attention and contribute to advancing this technology.'
                    : '加入我们的研究社区，探索 DeepSeek 稀疏注意力并为推进这项技术做出贡献。'
                  }
                </p>
                
                <div className="space-y-4 mb-6">
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-1">
                        {language === 'en' ? 'Write Detailed Blog Posts' : '撰写详细博客文章'}
                      </h3>
                      <p className="text-slate-300 text-sm">
                        {language === 'en' 
                          ? 'Create comprehensive tutorials, analysis, and explanations of the DeepSeek-V3.2-Exp paper'
                          : '创建关于 DeepSeek-V3.2-Exp 论文的综合教程、分析和解释'
                        }
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-1">
                        {language === 'en' ? 'Propose Research Questions' : '提出研究问题'}
                      </h3>
                      <p className="text-slate-300 text-sm">
                        {language === 'en' 
                          ? 'Identify open problems, limitations, and potential improvements for sparse attention mechanisms'
                          : '识别稀疏注意力机制的开放问题、局限性和潜在改进'
                        }
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-purple-400 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-1">
                        {language === 'en' ? 'Contribute to Research' : '为研究做出贡献'}
                      </h3>
                      <p className="text-slate-300 text-sm">
                        {language === 'en' 
                          ? 'Implement experiments, reproduce results, and explore new applications'
                          : '实施实验、重现结果并探索新应用'
                        }
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-col sm:flex-row gap-4">
                  <a 
                    href="https://github.com/Open-Superintelligence-Lab/deepseek-sparse-attention-research"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 text-white font-semibold rounded-xl hover:from-emerald-700 hover:to-teal-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-emerald-500/25"
                  >
                    <svg className="w-5 h-5 group-hover:rotate-12 transition-transform duration-300" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                    <span className="group-hover:translate-x-1 transition-transform duration-300">
                      {language === 'en' ? '🚀 Contribute to Research' : '🚀 为研究做出贡献'}
                    </span>
                  </a>
                  
                  <a 
                    href="https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group inline-flex items-center gap-2 px-6 py-3 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-blue-500 hover:text-blue-400 transition-all duration-300 hover:bg-blue-500/10"
                  >
                    <svg className="w-5 h-5 group-hover:rotate-12 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <span className="group-hover:translate-x-1 transition-transform duration-300">
                      {language === 'en' ? '📄 Read Paper' : '📄 阅读论文'}
                    </span>
                  </a>
                  
                  <a 
                    href="https://discord.gg/your-discord-invite"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group inline-flex items-center gap-2 px-6 py-3 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-purple-500 hover:text-purple-400 transition-all duration-300 hover:bg-purple-500/10"
                  >
                    <svg className="w-5 h-5 group-hover:rotate-12 transition-transform duration-300" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
                    </svg>
                    <span className="group-hover:translate-x-1 transition-transform duration-300">
                      {language === 'en' ? '💬 Join Us' : '💬 Join Us'}
                    </span>
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Interactive Technical Cards */}
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            {/* Problem Card */}
            <div className="group bg-gradient-to-br from-red-900/20 to-red-800/20 backdrop-blur-sm border border-red-600/30 rounded-xl p-6 hover:border-red-500/50 transition-all duration-300 hover:scale-105">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-red-600 rounded-lg flex items-center justify-center group-hover:rotate-12 transition-transform duration-300">
                  <span className="text-2xl">⚠️</span>
                </div>
                <h3 className="text-xl font-bold text-white">
                  {language === 'en' ? 'The Problem' : '问题所在'}
                </h3>
              </div>
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-slate-300">
                  <span className="text-red-400 font-mono text-lg">O(L²)</span>
                  <span className="text-sm">→</span>
                  <span className="text-slate-400 text-sm">
                    {language === 'en' ? 'Quadratic scaling' : '二次缩放'}
                  </span>
                </div>
                <p className="text-slate-300 text-sm leading-relaxed">
                  {language === 'en' 
                    ? 'Standard attention requires every token to look at all previous tokens, making long contexts extremely expensive.'
                    : '标准注意力要求每个 token 都查看所有先前的 token，使得长上下文极其昂贵。'
                  }
                </p>
              </div>
            </div>

            {/* Solution Card */}
            <div className="group bg-gradient-to-br from-green-900/20 to-green-800/20 backdrop-blur-sm border border-green-600/30 rounded-xl p-6 hover:border-green-500/50 transition-all duration-300 hover:scale-105">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-green-600 rounded-lg flex items-center justify-center group-hover:rotate-12 transition-transform duration-300">
                  <span className="text-2xl">⚡</span>
                </div>
                <h3 className="text-xl font-bold text-white">
                  {language === 'en' ? 'The Solution' : '解决方案'}
                </h3>
              </div>
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-slate-300">
                  <span className="text-green-400 font-mono text-lg">O(Lk)</span>
                  <span className="text-sm">→</span>
                  <span className="text-slate-400 text-sm">
                    {language === 'en' ? 'Nearly linear' : '近似线性'}
                  </span>
                </div>
                <p className="text-slate-300 text-sm leading-relaxed">
                  {language === 'en' 
                    ? 'Lightning Indexer intelligently selects only the most relevant tokens, dramatically reducing computational cost.'
                    : '闪电索引器智能选择最相关的 token，大幅降低计算成本。'
                  }
                </p>
              </div>
            </div>
          </div>

          {/* Performance Comparison */}
          <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-8 mb-8">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                <span className="text-3xl">📊</span>
                {language === 'en' ? 'Performance Comparison' : '性能对比'}
              </h2>
              <p className="text-slate-400">
                {language === 'en' 
                  ? 'See how DeepSeek-V3.2-Exp compares to traditional dense attention'
                  : '看看 DeepSeek-V3.2-Exp 与传统密集注意力的对比'
                }
              </p>
            </div>
            
            <div className="grid md:grid-cols-3 gap-6">
              {/* Traditional Model */}
              <div className="text-center p-6 bg-gradient-to-br from-slate-700/50 to-slate-600/50 rounded-lg border border-slate-500/30">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">🐌</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {language === 'en' ? 'Dense Attention' : '密集注意力'}
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="text-red-400 font-mono">O(L²)</div>
                  <div className="text-slate-300">
                    {language === 'en' ? 'High cost' : '高成本'}
                  </div>
                  <div className="text-slate-300">
                    {language === 'en' ? 'Slow scaling' : '缩放缓慢'}
                  </div>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex items-center justify-center">
                <div className="text-4xl text-slate-400 animate-pulse">→</div>
              </div>

              {/* Sparse Model */}
              <div className="text-center p-6 bg-gradient-to-br from-emerald-700/50 to-emerald-600/50 rounded-lg border border-emerald-500/30">
                <div className="w-16 h-16 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">🚀</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {language === 'en' ? 'Sparse Attention' : '稀疏注意力'}
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="text-emerald-400 font-mono">O(Lk)</div>
                  <div className="text-slate-300">
                    {language === 'en' ? 'Low cost' : '低成本'}
                  </div>
                  <div className="text-slate-300">
                    {language === 'en' ? 'Fast scaling' : '快速缩放'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Key Points Summary */}
          <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-8 mb-8">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                <span className="text-3xl">🔬</span>
                {language === 'en' ? 'Key Technical Insights' : '关键技术洞察'}
              </h2>
              <p className="text-slate-400">
                {language === 'en' 
                  ? 'Essential concepts from the DeepSeek-V3.2-Exp research paper'
                  : 'DeepSeek-V3.2-Exp 研究论文的核心概念'
                }
              </p>
            </div>
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
