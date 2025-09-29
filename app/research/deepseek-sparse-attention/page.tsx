'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function DeepSeekSparseAttentionPage() {
  const { language } = useLanguage();

  const project = language === 'en' ? {
    title: "DeepSeek V3.2-Exp Implementation",
    description: "Implementing DeepSeek V3.2-Exp architecture with sparse attention mechanisms for efficient long-context processing",
    status: "In Progress",
    features: [
      "DeepSeek Sparse Attention (DSA)",
      "Lightning Indexer implementation",
      "Mixture-of-Latent-Attention (MLA)",
      "128K context window support",
      "Memory optimization techniques"
    ],
    details: "This project aims to implement the DeepSeek V3.2-Exp architecture as described in the research paper. The implementation focuses on the novel DeepSeek Sparse Attention (DSA) mechanism that enables efficient processing of up to 128K tokens while maintaining computational efficiency. The task involves implementing the Lightning Indexer for token selection and the Mixture-of-Latent-Attention (MLA) for memory optimization.",
    task: "Implement the DeepSeek V3.2-Exp architecture from the research paper: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf",
    researchQuestions: [
      "How can we optimize the Lightning Indexer algorithm for different sequence lengths and task types?",
      "What are the optimal sparse attention patterns for maintaining performance across diverse long-context tasks?",
      "How can we adaptively adjust the token selection parameter k based on context complexity and computational budget?"
    ],
    tasks: [
      "Review and understand DeepGEMM PR #200: https://github.com/deepseek-ai/DeepGEMM/pull/200",
      "Review and understand FlashMLA PR #98: https://github.com/deepseek-ai/FlashMLA/pull/98",
      "Create markdown explanations and practice exercises for both",
      "Create YouTube videos explaining both",
      "Create paid Skool bonus exercises",
      "Translate all materials to Chinese"
    ]
  } : {
    title: "DeepSeek V3.2-Exp 实现",
    description: "实现DeepSeek V3.2-Exp架构，采用稀疏注意力机制进行高效长上下文处理",
    status: "进行中",
    features: [
      "DeepSeek稀疏注意力(DSA)",
      "闪电索引器实现",
      "潜在注意力混合(MLA)",
      "128K上下文窗口支持",
      "内存优化技术"
    ],
    details: "本项目旨在实现研究论文中描述的DeepSeek V3.2-Exp架构。实现重点在于新颖的DeepSeek稀疏注意力(DSA)机制，能够在保持计算效率的同时高效处理多达128K个token。任务包括实现用于token选择的闪电索引器和用于内存优化的潜在注意力混合(MLA)。",
    task: "实现研究论文中的DeepSeek V3.2-Exp架构：https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf",
    researchQuestions: [
      "如何针对不同序列长度和任务类型优化闪电索引器算法？",
      "在多样化的长上下文任务中，保持性能的最佳稀疏注意力模式是什么？",
      "如何根据上下文复杂度和计算预算自适应调整token选择参数k？"
    ],
    tasks: [
      "审查和理解DeepGEMM PR #200: https://github.com/deepseek-ai/DeepGEMM/pull/200",
      "审查和理解FlashMLA PR #98: https://github.com/deepseek-ai/FlashMLA/pull/98",
      "为两者创建markdown解释和实践练习",
      "创建YouTube视频解释两者",
      "创建付费Skool奖励练习",
      "将所有材料翻译成中文"
    ]
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

            {/* Task Section */}
            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-4 text-blue-400">
                {language === 'en' ? 'Implementation Task' : '实现任务'}
              </h3>
              <div className="bg-slate-800/50 border border-slate-600/50 rounded-lg p-4 mb-6">
                <p className="text-gray-300 leading-relaxed">
                  {project.task}
                </p>
              </div>
            </div>

            {/* Features Section */}
            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-4 text-blue-400">
                {language === 'en' ? 'Key Components' : '主要组件'}
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

            {/* Research Questions Section */}
            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-4 text-blue-400">
                {language === 'en' ? 'Research Questions' : '研究问题'}
              </h3>
              <div className="space-y-4">
                {project.researchQuestions.map((question, index) => (
                  <div key={index} className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <span className="bg-blue-500/20 text-blue-400 text-sm px-2 py-1 rounded-md font-medium">
                        {index + 1}
                      </span>
                      <p className="text-gray-300 leading-relaxed">{question}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Tasks Section */}
            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-4 text-blue-400">
                {language === 'en' ? 'Tasks' : '任务'}
              </h3>
              <div className="space-y-3">
                {project.tasks.map((task, index) => (
                  <div key={index} className="flex items-start gap-3 text-gray-300">
                    <div className="w-2 h-2 bg-blue-400 rounded-full mt-2"></div>
                    <span className="leading-relaxed">{task}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500">DeepSeek V3.2-Exp Implementation</span>
              <div className="flex gap-3">
                <a 
                  href="https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500/20 to-blue-600/20 border border-blue-500/50 rounded-lg text-blue-400 hover:bg-blue-500/30 hover:border-blue-400/70 transition-all duration-200"
                >
                  <span>{language === 'en' ? 'Research Paper' : '研究论文'}</span>
                  <span>↗</span>
                </a>
                <a 
                  href="https://github.com/deepseek-ai/DeepSeek-V3.2-Exp"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-slate-600/20 to-slate-700/20 border border-slate-500/50 rounded-lg text-slate-300 hover:bg-slate-600/30 hover:border-slate-400/70 transition-all duration-200"
                >
                  <span>{language === 'en' ? 'GitHub' : 'GitHub'}</span>
                  <span>↗</span>
                </a>
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
