'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function TrainLLMFor1DollarPage() {
  const { language } = useLanguage();

  const project = language === 'en' ? {
    title: "Train LLM For $1",
    description: "Revolutionary research on ultra-low-cost large language model training using innovative optimization techniques and distributed computing strategies",
    status: "Research",
    features: [
      "Ultra-low-cost training methodologies",
      "Distributed computing optimization",
      "Memory-efficient algorithms",
      "Open-source implementation",
      "Scalable architecture design"
    ],
    details: "This groundbreaking research explores how to train state-of-the-art large language models for just $1 using advanced optimization techniques, distributed computing strategies, and memory-efficient algorithms. Our approach combines innovative model architectures with cost-effective training methodologies to democratize AI development."
  } : {
    title: "1美元训练大模型",
    description: "使用创新优化技术和分布式计算策略进行超低成本大语言模型训练的革命性研究",
    status: "研究",
    features: [
      "超低成本训练方法",
      "分布式计算优化",
      "内存高效算法",
      "开源实现",
      "可扩展架构设计"
    ],
    details: "这项突破性研究探索如何使用先进的优化技术、分布式计算策略和内存高效算法，仅用1美元训练最先进的大语言模型。我们的方法将创新的模型架构与成本效益高的训练方法相结合，以民主化AI开发。"
  };

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden" aria-labelledby="hero-title">
        <div className="absolute inset-0 bg-gradient-to-r from-green-600/20 via-emerald-600/20 to-green-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-green-500/5 to-transparent"></div>
        </div>
        
        <div className="relative container mx-auto px-4 sm:px-6 py-16 sm:py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 id="hero-title" className="text-4xl sm:text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
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
          <article className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 hover:border-green-500/50 hover:shadow-2xl hover:shadow-green-500/10 transition-all duration-300">
            <div className="flex items-start justify-between mb-6">
              <div className="flex items-center gap-3">
                <span className="bg-slate-600/50 text-slate-300 text-xs px-3 py-1 rounded-md">Research</span>
                <span className="bg-green-500/20 text-green-400 text-xs px-3 py-1 rounded-md">{project.status}</span>
              </div>
            </div>
            
            <div className="mb-6">
              <h2 className="text-3xl font-bold mb-4 group-hover:text-green-400 transition-colors">
                {project.title}
              </h2>
              <p className="text-gray-300 text-lg leading-relaxed mb-6">
                {project.details}
              </p>
            </div>

            {/* Features Section */}
            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-4 text-green-400">
                {language === 'en' ? 'Key Features' : '主要特性'}
              </h3>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {project.features.map((feature, index) => (
                  <li key={index} className="flex items-center gap-3 text-gray-300">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500">Open Superintelligence Lab</span>
              <div className="flex gap-3">
                <button className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-500/20 to-green-600/20 border border-green-500/50 rounded-lg text-green-400 hover:bg-green-500/30 hover:border-green-400/70 transition-all duration-200">
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
