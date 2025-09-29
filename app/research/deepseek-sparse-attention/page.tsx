'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function DeepSeekSparseAttentionPage() {
  const { language } = useLanguage();

  const project = language === 'en' ? {
    title: "DeepSeek Sparse Attention",
    description: "Advanced research on DeepSeek's innovative sparse attention mechanisms for efficient long-context processing and memory optimization",
    status: "Open Source",
    features: [
      "Sparse attention mechanisms",
      "Long-context efficiency",
      "Memory optimization",
      "V3.2-Exp architecture",
      "Open-source implementation"
    ],
    details: "This research focuses on DeepSeek's breakthrough sparse attention technology that enables efficient processing of long contexts while maintaining high performance. The DeepSeek Sparse Attention (DSA) mechanism significantly reduces computational complexity and memory requirements, making it possible to handle much longer sequences than traditional attention mechanisms."
  } : {
    title: "DeepSeek稀疏注意力",
    description: "DeepSeek创新稀疏注意力机制的高效长上下文处理和内存优化研究",
    status: "开源",
    features: [
      "稀疏注意力机制",
      "长上下文效率",
      "内存优化",
      "V3.2-Exp架构",
      "开源实现"
    ],
    details: "这项研究专注于DeepSeek突破性的稀疏注意力技术，能够在保持高性能的同时实现长上下文的高效处理。DeepSeek稀疏注意力(DSA)机制显著降低了计算复杂度和内存需求，使其能够处理比传统注意力机制更长的序列。"
  };

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden" aria-labelledby="hero-title">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
        </div>
        
        <div className="relative container mx-auto px-4 sm:px-6 py-16 sm:py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 id="hero-title" className="text-4xl sm:text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              {project.title}
            </h1>
            <p className="text-lg sm:text-xl text-gray-300 mb-8">
              {project.description}
            </p>
          </div>
        </div>
      </section>

      <main className="container mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <div className="max-w-4xl mx-auto">
          <article className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 hover:border-blue-500/50 hover:shadow-2xl hover:shadow-blue-500/10 transition-all duration-300">
            <div className="flex items-start justify-between mb-6">
              <div className="flex items-center gap-3">
                <span className="bg-slate-600/50 text-slate-300 text-xs px-3 py-1 rounded-md">Research</span>
                <span className="bg-blue-500/20 text-blue-400 text-xs px-3 py-1 rounded-md">{project.status}</span>
              </div>
            </div>
            
            <div className="mb-6">
              <h2 className="text-3xl font-bold mb-4 group-hover:text-blue-400 transition-colors">
                {project.title}
              </h2>
              <p className="text-gray-300 text-lg leading-relaxed mb-6">
                {project.details}
              </p>
            </div>

            {/* Features Section */}
            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-4 text-blue-400">
                {language === 'en' ? 'Key Features' : '主要特性'}
              </h3>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {project.features.map((feature, index) => (
                  <li key={index} className="flex items-center gap-3 text-gray-300">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500">DeepSeek Research</span>
              <div className="flex gap-3">
                <button className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500/20 to-blue-600/20 border border-blue-500/50 rounded-lg text-blue-400 hover:bg-blue-500/30 hover:border-blue-400/70 transition-all duration-200">
                  <span>{language === 'en' ? 'View Research' : '查看研究'}</span>
                  <span>→</span>
                </button>
                <button className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-slate-600/20 to-slate-700/20 border border-slate-500/50 rounded-lg text-slate-300 hover:bg-slate-600/30 hover:border-slate-400/70 transition-all duration-200">
                  <span>{language === 'en' ? 'GitHub' : 'GitHub'}</span>
                  <span>↗</span>
                </button>
              </div>
            </div>
          </article>
        </div>

        {/* Back to Home */}
        <div className="text-center mt-12 sm:mt-16">
          <Link 
            href="/" 
            className="inline-flex items-center gap-2 px-4 sm:px-6 py-2 sm:py-3 bg-gradient-to-r from-slate-700/50 to-slate-600/50 border border-slate-500/50 rounded-lg hover:border-blue-500/50 hover:bg-slate-600/50 transition-all duration-200 text-sm sm:text-base"
          >
            <span>←</span>
            <span>{language === 'en' ? 'Back to Home' : '返回首页'}</span>
          </Link>
        </div>
      </main>
    </>
  );
}
