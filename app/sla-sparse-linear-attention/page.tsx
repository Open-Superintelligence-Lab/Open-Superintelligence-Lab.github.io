'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function SLASparseLinearAttention() {
  const { language } = useLanguage();

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Background effects */}
        <div className="absolute inset-0 bg-gradient-to-r from-orange-600/20 via-red-600/20 to-pink-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-orange-500/5 to-transparent"></div>
        </div>
        
        {/* Animated background particles */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/6 left-1/6 w-3 h-3 bg-gradient-to-r from-orange-400 to-red-400 rounded-full opacity-60 animate-pulse"></div>
          <div className="absolute top-1/4 right-1/5 w-2 h-2 bg-gradient-to-r from-red-400 to-pink-400 rounded-full opacity-50 animate-pulse delay-300"></div>
          <div className="absolute top-1/3 left-1/8 w-4 h-4 bg-gradient-to-r from-pink-400 to-orange-400 rounded-full opacity-40 animate-pulse delay-700"></div>
          <div className="absolute bottom-1/4 right-1/6 w-2.5 h-2.5 bg-gradient-to-r from-orange-400 to-red-400 rounded-full opacity-55 animate-pulse delay-1000"></div>
        </div>
        
        <div className="relative container mx-auto px-6 pt-32 pb-24">
          <div className="text-center max-w-4xl mx-auto">
            <div className="relative">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium mb-8 leading-tight">
                <span className="bg-gradient-to-r from-orange-400 via-red-400 to-pink-400 bg-clip-text text-transparent">
                  {language === 'en' ? 'SLA: Sparse-Linear Attention' : 'SLA: 稀疏线性注意力'}
                </span>
              </h1>
              
              {/* Glow effect for the title */}
              <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-medium leading-tight blur-sm">
                <span className="bg-gradient-to-r from-orange-400/20 via-red-400/20 to-pink-400/20 bg-clip-text text-transparent">
                  {language === 'en' ? 'SLA: Sparse-Linear Attention' : 'SLA: 稀疏线性注意力'}
                </span>
              </div>
            </div>
            
            <p className="text-xl text-slate-300 mb-12 leading-relaxed">
              {language === 'en' 
                ? 'Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention'
                : '通过可微调稀疏线性注意力超越扩散变换器中的稀疏性'
              }
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
        <div className="container mx-auto px-6 max-w-6xl">
          
          {/* Research Paper Section */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0">
                <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-red-500 rounded-xl flex items-center justify-center">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-bold text-white mb-4">
                  {language === 'en' ? 'Research Paper' : '研究论文'}
                </h2>
                <p className="text-slate-300 mb-6 leading-relaxed">
                  {language === 'en' 
                    ? 'SLA (Sparse-Linear Attention) is a trainable attention method that fuses sparse and linear attention to accelerate diffusion models. It achieves a 20x reduction in attention computation with minimal quality loss.'
                    : 'SLA（稀疏线性注意力）是一种可训练的注意力方法，融合稀疏注意力和线性注意力来加速扩散模型。它在最小质量损失的情况下实现了20倍的注意力计算减少。'
                  }
                </p>
                <div className="flex flex-col sm:flex-row gap-4">
                  <a 
                    href="https://arxiv.org/abs/2509.24006"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-orange-600 to-red-600 text-white font-semibold rounded-xl hover:from-orange-700 hover:to-red-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-orange-500/25"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    {language === 'en' ? 'Read Paper' : '阅读论文'}
                  </a>
                  <a 
                    href="https://github.com/thu-ml/SLA"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-slate-600 to-slate-700 text-white font-semibold rounded-xl hover:from-slate-700 hover:to-slate-800 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-slate-500/25"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                    {language === 'en' ? 'View Code' : '查看代码'}
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Key Features */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 bg-orange-500/20 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-white">
                  {language === 'en' ? '20x Speedup' : '20倍加速'}
                </h3>
              </div>
              <p className="text-slate-300 text-sm leading-relaxed">
                {language === 'en' 
                  ? 'Achieves 20x reduction in attention computation with minimal quality loss in diffusion models.'
                  : '在扩散模型中实现20倍的注意力计算减少，质量损失最小。'
                }
              </p>
            </div>

            <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 bg-red-500/20 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-white">
                  {language === 'en' ? '95% Reduction' : '95%减少'}
                </h3>
              </div>
              <p className="text-slate-300 text-sm leading-relaxed">
                {language === 'en' 
                  ? 'Reduces attention computation by 95% without degrading end-to-end generation quality.'
                  : '在不降低端到端生成质量的情况下，将注意力计算减少95%。'
                }
              </p>
            </div>

            <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 bg-pink-500/20 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-pink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-white">
                  {language === 'en' ? 'GPU Kernel' : 'GPU内核'}
                </h3>
              </div>
              <p className="text-slate-300 text-sm leading-relaxed">
                {language === 'en' 
                  ? 'Efficient GPU kernel implementation yields 13.7x speedup in attention computation.'
                  : '高效的GPU内核实现使注意力计算加速13.7倍。'
                }
              </p>
            </div>
          </div>

          {/* Technical Details */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">
              {language === 'en' ? 'Technical Innovation' : '技术创新'}
            </h2>
            
            <div className="space-y-6">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 bg-orange-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                  <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {language === 'en' ? 'Attention Weight Classification' : '注意力权重分类'}
                  </h3>
                  <p className="text-slate-300 leading-relaxed">
                    {language === 'en' 
                      ? 'SLA classifies attention weights into critical, marginal, and negligible categories, applying O(N²) attention to critical weights, O(N) attention to marginal weights, and skipping negligible ones.'
                      : 'SLA将注意力权重分为关键、边缘和可忽略三类，对关键权重应用O(N²)注意力，对边缘权重应用O(N)注意力，跳过可忽略的权重。'
                    }
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-8 h-8 bg-red-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                  <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {language === 'en' ? 'Hybrid Approach' : '混合方法'}
                  </h3>
                  <p className="text-slate-300 leading-relaxed">
                    {language === 'en' 
                      ? 'Combines sparse acceleration for high-rank weights with low-rank acceleration for remaining weights, fusing both computations into a single GPU kernel.'
                      : '将高秩权重的稀疏加速与剩余权重的低秩加速相结合，将两种计算融合到单个GPU内核中。'
                    }
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-8 h-8 bg-pink-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                  <svg className="w-5 h-5 text-pink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {language === 'en' ? 'Video Generation Focus' : '视频生成重点'}
                  </h3>
                  <p className="text-slate-300 leading-relaxed">
                    {language === 'en' 
                      ? 'Specifically designed for Diffusion Transformer models in video generation, where attention latency is a major bottleneck due to long sequence lengths and quadratic complexity.'
                      : '专门为视频生成中的扩散变换器模型设计，由于长序列长度和二次复杂度，注意力延迟是主要瓶颈。'
                    }
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Performance Results */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">
              {language === 'en' ? 'Performance Results' : '性能结果'}
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="text-center">
                <div className="w-20 h-20 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-white">20x</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {language === 'en' ? 'Attention Reduction' : '注意力减少'}
                </h3>
                <p className="text-slate-300 text-sm">
                  {language === 'en' 
                    ? 'Reduction in attention computation'
                    : '注意力计算减少'
                  }
                </p>
              </div>

              <div className="text-center">
                <div className="w-20 h-20 bg-gradient-to-r from-red-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-white">13.7x</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {language === 'en' ? 'GPU Speedup' : 'GPU加速'}
                </h3>
                <p className="text-slate-300 text-sm">
                  {language === 'en' 
                    ? 'Speedup in attention computation'
                    : '注意力计算加速'
                  }
                </p>
              </div>

              <div className="text-center">
                <div className="w-20 h-20 bg-gradient-to-r from-pink-500 to-orange-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-white">2.2x</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {language === 'en' ? 'End-to-End' : '端到端'}
                </h3>
                <p className="text-slate-300 text-sm">
                  {language === 'en' 
                    ? 'Speedup in video generation'
                    : '视频生成加速'
                  }
                </p>
              </div>

              <div className="text-center">
                <div className="w-20 h-20 bg-gradient-to-r from-orange-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-white">95%</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {language === 'en' ? 'Computation Cut' : '计算削减'}
                </h3>
                <p className="text-slate-300 text-sm">
                  {language === 'en' 
                    ? 'Reduction without quality loss'
                    : '无质量损失的减少'
                  }
                </p>
              </div>
            </div>
          </div>

          {/* Back to Home */}
          <div className="text-center">
            <Link 
              href="/"
              className="inline-flex items-center gap-2 px-6 py-3 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-orange-500 hover:text-orange-400 transition-all duration-300 transform hover:scale-105"
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
