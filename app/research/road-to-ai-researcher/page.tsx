'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function RoadToAIResearcherPage() {
  const { language } = useLanguage();

  const project = language === 'en' ? {
    title: "Road to AI Researcher",
    description: "A comprehensive journey into becoming an AI researcher, covering everything from foundational concepts to cutting-edge research methodologies",
    status: "Learning Path",
    features: [
      "Comprehensive AI fundamentals",
      "Research methodology training",
      "Hands-on project experience",
      "Open-source contributions",
      "Community-driven learning"
    ],
    details: "This comprehensive learning path guides aspiring AI researchers through the essential skills, knowledge, and experiences needed to become successful in the field. From foundational concepts to advanced research techniques, this program provides a structured approach to building expertise in artificial intelligence research."
  } : {
    title: "AI研究员之路",
    description: "成为AI研究员的综合学习路径，涵盖从基础概念到前沿研究方法的所有内容",
    status: "学习路径",
    features: [
      "全面的AI基础知识",
      "研究方法培训",
      "实践项目经验",
      "开源贡献",
      "社区驱动学习"
    ],
    details: "这个综合学习路径指导有抱负的AI研究员掌握在该领域取得成功所需的基本技能、知识和经验。从基础概念到高级研究技术，这个项目提供了构建人工智能研究专业知识的结构化方法。"
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
                <span className="bg-slate-600/50 text-slate-300 text-xs px-3 py-1 rounded-md">Learning Path</span>
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
                <a 
                  href="https://github.com/vukrosic/blueberry-llm-kimi-deepseek"
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
