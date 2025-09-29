'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function DeepSeekResearchPage() {
  const { language } = useLanguage();

  const project = language === 'en' ? {
    title: "V3.2-Exp Architecture",
    description: "Open source research on DeepSeek Sparse Attention (DSA) and long-context efficiency improvements",
    status: "Open Source",
    href: "/research/deepseek/v3-2-exp",
  } : {
    title: "V3.2-Exp架构",
    description: "DeepSeek稀疏注意力(DSA)和长上下文效率改进的开源研究",
    status: "开源",
    href: "/research/deepseek/v3-2-exp",
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
              DeepSeek Research
            </h1>
            <p className="text-lg sm:text-xl text-gray-300 mb-8">
              {language === 'en' 
                ? "Comprehensive research hub for DeepSeek's latest AI breakthroughs and open-source innovations"
                : "DeepSeek最新AI突破和开源创新的综合研究平台"
              }
            </p>
          </div>
        </div>
      </section>

      <main className="container mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <div className="max-w-4xl mx-auto">
          <article className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 hover:border-orange-500/50 hover:shadow-2xl hover:shadow-orange-500/10 transition-all duration-300">
            <div className="flex items-start justify-between mb-6">
              <div className="flex items-center gap-3">
                <span className="bg-slate-600/50 text-slate-300 text-xs px-3 py-1 rounded-md">Research</span>
                <span className="bg-orange-500/20 text-orange-400 text-xs px-3 py-1 rounded-md">{project.status}</span>
              </div>
            </div>
            
            <div className="mb-6">
              <h2 className="text-3xl font-bold mb-4 group-hover:text-orange-400 transition-colors">
                {project.title}
              </h2>
              <p className="text-gray-300 text-lg leading-relaxed mb-6">
                {project.description}
              </p>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500">DeepSeek Research</span>
              <Link 
                href={project.href}
                className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-orange-500/20 to-orange-600/20 border border-orange-500/50 rounded-lg text-orange-400 hover:bg-orange-500/30 hover:border-orange-400/70 transition-all duration-200"
              >
                <span>{language === 'en' ? 'Explore Research' : '探索研究'}</span>
                <span>→</span>
              </Link>
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
