'use client';

import Link from "next/link";
import { useState } from "react";
import { Navigation } from "@/components/navigation";
import { useLanguage } from "@/components/providers/language-provider";
import { translations } from "@/lib/language-detection";

export default function DeepSeekResearchPage() {
  const { language } = useLanguage();
  const t = translations[language];

  // State for managing published status
  const [publishedProjects, setPublishedProjects] = useState({
    "visual-hallucinations": true,
    "healthcare-applications": true,
    "information-suppression": false,
    "reasoning-replication": true,
    "v3-2-exp": true
  });

  const togglePublish = (projectId: string) => {
    setPublishedProjects(prev => ({
      ...prev,
      [projectId]: !prev[projectId as keyof typeof prev]
    }));
  };

  const researchProjects = language === 'en' ? [
    {
      id: "visual-hallucinations",
      title: "Visual Hallucination Vulnerabilities",
      description: "Research on inducing targeted visual hallucinations in multimodal LLMs through embedding manipulation attacks",
      status: "Active",
      statusColor: "green",
      href: "/research/deepseek/visual-hallucinations",
    },
    {
      id: "healthcare-applications",
      title: "Healthcare AI Applications",
      description: "Survey of DeepSeek-R1 capabilities, risks, and clinical applications in healthcare diagnostics",
      status: "Research",
      statusColor: "blue",
      href: "/research/deepseek/healthcare",
    },
    {
      id: "information-suppression",
      title: "Information Suppression Audit",
      description: "Auditing and quantifying censorship patterns in DeepSeek models for transparency analysis",
      status: "Analysis",
      statusColor: "purple",
      href: "/research/deepseek/information-suppression",
    },
    {
      id: "reasoning-replication",
      title: "Reasoning Model Replication",
      description: "100-day replication study of DeepSeek-R1 training procedures and reasoning capabilities",
      status: "Complete",
      statusColor: "cyan",
      href: "/research/deepseek/reasoning-replication",
    },
    {
      id: "v3-2-exp",
      title: "V3.2-Exp Architecture",
      description: "Open source research on DeepSeek Sparse Attention (DSA) and long-context efficiency improvements",
      status: "Open Source",
      statusColor: "orange",
      href: "/research/deepseek/v3-2-exp",
    }
  ] : [
    {
      id: "visual-hallucinations",
      title: "视觉幻觉漏洞",
      description: "通过嵌入操作攻击在多模态大语言模型中诱导目标视觉幻觉的研究",
      status: "活跃",
      statusColor: "green",
      href: "/research/deepseek/visual-hallucinations",
    },
    {
      id: "healthcare-applications",
      title: "医疗AI应用",
      description: "DeepSeek-R1在医疗诊断中的能力、风险和临床应用调查",
      status: "研究",
      statusColor: "blue",
      href: "/research/deepseek/healthcare",
    },
    {
      id: "information-suppression",
      title: "信息抑制审计",
      description: "审计和量化DeepSeek模型中的审查模式，用于透明度分析",
      status: "分析",
      statusColor: "purple",
      href: "/research/deepseek/information-suppression",
    },
    {
      id: "reasoning-replication",
      title: "推理模型复制",
      description: "DeepSeek-R1训练程序和推理能力的100天复制研究",
      status: "完成",
      statusColor: "cyan",
      href: "/research/deepseek/reasoning-replication",
    },
    {
      id: "v3-2-exp",
      title: "V3.2-Exp架构",
      description: "DeepSeek稀疏注意力(DSA)和长上下文效率改进的开源研究",
      status: "开源",
      statusColor: "orange",
      href: "/research/deepseek/v3-2-exp",
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <Navigation />
      
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
        </div>
        
        <div className="relative container mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              DeepSeek Research
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              {language === 'en' 
                ? "Comprehensive research hub for DeepSeek's latest AI breakthroughs and open-source innovations"
                : "DeepSeek最新AI突破和开源创新的综合研究平台"
              }
            </p>
          </div>
        </div>
      </section>

      <main className="container mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h3 className="text-3xl font-bold mb-4">
            {language === 'en' ? 'Research Projects' : '研究项目'}
          </h3>
          <p className="text-gray-400 mb-6">
            {language === 'en' 
              ? 'Explore cutting-edge DeepSeek research across multiple domains'
              : '探索DeepSeek在多个领域的前沿研究'
            }
          </p>
          
          {/* Filter Options */}
          <div className="flex justify-center gap-4 mb-8">
            <button className="px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg hover:border-blue-500/50 transition-colors">
              {language === 'en' ? 'All Projects' : '所有项目'}
            </button>
            <button className="px-4 py-2 bg-green-500/20 border border-green-500/50 rounded-lg text-green-400 hover:bg-green-500/30 transition-colors">
              {language === 'en' ? 'Published' : '已发布'}
            </button>
            <button className="px-4 py-2 bg-gray-500/20 border border-gray-500/50 rounded-lg text-gray-400 hover:bg-gray-500/30 transition-colors">
              {language === 'en' ? 'Drafts' : '草稿'}
            </button>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
          {researchProjects.map((project) => {
            const isPublished = publishedProjects[project.id as keyof typeof publishedProjects];
            return (
            <div key={project.id} className="relative">
              <Link 
                href={project.href}
                className={`group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-${project.statusColor}-500/50 hover:shadow-2xl hover:shadow-${project.statusColor}-500/10 transition-all duration-300 ${!isPublished ? 'opacity-60' : ''}`}
              >
                <div className="absolute top-4 left-4">
                  <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">Research</span>
                </div>
                <div className={`absolute top-4 right-4`}>
                  <span className={`bg-${project.statusColor}-500/20 text-${project.statusColor}-400 text-xs px-2 py-1 rounded-md`}>
                    {project.status}
                  </span>
                </div>
                
                <div className="mt-8">
                  <h4 className={`text-xl font-bold mb-3 group-hover:text-${project.statusColor}-400 transition-colors`}>
                    {project.title}
                  </h4>
                  <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                    {project.description}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">DeepSeek Research</span>
                    <span className={`text-${project.statusColor}-400 text-sm group-hover:text-${project.statusColor}-300 transition-colors`}>
                      {language === 'en' ? 'Explore →' : '探索 →'}
                    </span>
                  </div>
                </div>
              </Link>
              
              {/* Publish/Unpublish Toggle */}
              <div className="absolute bottom-4 right-4">
                <button
                  className={`text-xs px-2 py-1 rounded-md transition-colors ${
                    isPublished 
                      ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30' 
                      : 'bg-gray-500/20 text-gray-400 hover:bg-gray-500/30'
                  }`}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    togglePublish(project.id);
                  }}
                >
                  {isPublished 
                    ? (language === 'en' ? 'Published' : '已发布') 
                    : (language === 'en' ? 'Draft' : '草稿')
                  }
                </button>
              </div>
            </div>
            );
          })}
        </div>

        {/* Back to Home */}
        <div className="text-center mt-16">
          <Link 
            href="/" 
            className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-slate-700/50 to-slate-600/50 border border-slate-500/50 rounded-lg hover:border-blue-500/50 hover:bg-slate-600/50 transition-all duration-200"
          >
            <span>←</span>
            <span>{language === 'en' ? 'Back to Home' : '返回首页'}</span>
          </Link>
        </div>
      </main>
    </div>
  );
}
