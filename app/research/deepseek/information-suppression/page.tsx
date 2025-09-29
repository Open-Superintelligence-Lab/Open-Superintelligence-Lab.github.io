'use client';

import Link from "next/link";
import { Navigation } from "@/components/navigation";
import { useLanguage } from "@/components/providers/language-provider";

export default function InformationSuppressionPage() {
  const { language } = useLanguage();

  const researchQuestions = language === 'en' ? [
    "How can an effective framework be designed to audit LLMs for content suppression?",
    "What techniques can detect semantic-level information suppression within model outputs?",
    "How can LLMs be engineered to ensure transparency and minimize unintended censorship?",
    "What are the patterns of information suppression across different political and cultural contexts?",
    "How do suppression mechanisms vary between different model architectures and training approaches?",
    "What metrics can quantitatively measure the extent and nature of information suppression?",
    "How can we distinguish between intentional safety measures and problematic censorship?",
    "What are the implications of information suppression for democratic discourse and academic freedom?"
  ] : [
    "如何设计一个有效的框架来审计大语言模型的内容抑制？",
    "什么技术可以检测模型输出中的语义级信息抑制？",
    "如何设计大语言模型以确保透明度并最小化无意的审查？",
    "信息抑制在不同政治和文化背景下的模式是什么？",
    "抑制机制在不同模型架构和训练方法之间如何变化？",
    "什么指标可以定量测量信息抑制的程度和性质？",
    "我们如何区分有意的安全措施和有问题的审查？",
    "信息抑制对民主话语和学术自由的影响是什么？"
  ];

  const researchAreas = language === 'en' ? [
    {
      title: "Auditing Framework Development",
      description: "Design and implement comprehensive frameworks for auditing LLMs to identify and quantify content suppression patterns."
    },
    {
      title: "Semantic Analysis",
      description: "Develop advanced techniques for detecting semantic-level information suppression within model outputs and responses."
    },
    {
      title: "Transparency Enhancement",
      description: "Research methods to improve model transparency and reduce unintended censorship while maintaining safety standards."
    },
    {
      title: "Quantitative Metrics",
      description: "Create robust metrics for measuring suppression rates, bias patterns, and content accessibility across different topics."
    }
  ] : [
    {
      title: "审计框架开发",
      description: "设计和实施全面的框架来审计大语言模型，识别和量化内容抑制模式。"
    },
    {
      title: "语义分析",
      description: "开发先进技术来检测模型输出和响应中的语义级信息抑制。"
    },
    {
      title: "透明度增强",
      description: "研究方法以提高模型透明度并减少无意审查，同时保持安全标准。"
    },
    {
      title: "定量指标",
      description: "创建强大的指标来测量不同主题的抑制率、偏见模式和内容可访问性。"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <Navigation />
      
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 via-pink-600/20 to-purple-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-purple-500/5 to-transparent"></div>
        </div>
        
        <div className="relative container mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Information Suppression Audit
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              {language === 'en' 
                ? "Auditing and quantifying censorship patterns in DeepSeek models for transparency analysis"
                : "审计和量化DeepSeek模型中的审查模式，用于透明度分析"
              }
            </p>
            <div className="flex justify-center gap-4">
              <span className="bg-purple-500/20 text-purple-400 text-sm px-3 py-1 rounded-md">Transparency</span>
              <span className="bg-pink-500/20 text-pink-400 text-sm px-3 py-1 rounded-md">Audit Research</span>
            </div>
          </div>
        </div>
      </section>

      <main className="container mx-auto px-6 py-12">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <Link 
              href="/research/deepseek" 
              className="text-gray-400 hover:text-white transition-colors"
            >
              ← Back to DeepSeek Research
            </Link>
          </div>

          {/* Research Overview */}
          <section className="mb-12">
            <h2 className="text-3xl font-bold mb-6">
              {language === 'en' ? 'Research Overview' : '研究概述'}
            </h2>
            <div className="prose prose-invert max-w-none">
              <p className="text-lg text-gray-300 mb-6">
                {language === 'en' 
                  ? "This research audits DeepSeek's LLMs to identify and quantify instances of information suppression, particularly concerning politically sensitive content. The study develops comprehensive auditing frameworks to detect both explicit and subtle forms of content filtering, providing insights into how models handle controversial topics and potentially sensitive information."
                  : "这项研究审计了DeepSeek的大语言模型，以识别和量化信息抑制的实例，特别是涉及政治敏感内容的情况。研究开发了全面的审计框架来检测显式和微妙的内容过滤形式，提供关于模型如何处理争议话题和潜在敏感信息的见解。"
                }
              </p>
              
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/30">
                <h3 className="text-xl font-semibold mb-3 text-purple-400">
                  {language === 'en' ? 'Research Objectives' : '研究目标'}
                </h3>
                <ul className="space-y-2 text-gray-300">
                  <li>• {language === 'en' ? 'Develop systematic auditing methodologies for content suppression' : '开发内容抑制的系统性审计方法'}</li>
                  <li>• {language === 'en' ? 'Quantify suppression patterns across different topic categories' : '量化不同主题类别的抑制模式'}</li>
                  <li>• {language === 'en' ? 'Identify semantic-level filtering mechanisms' : '识别语义级过滤机制'}</li>
                  <li>• {language === 'en' ? 'Assess transparency and accountability in AI systems' : '评估AI系统的透明度和问责制'}</li>
                </ul>
              </div>
            </div>
          </section>

          {/* Research Questions */}
          <section className="mb-12">
            <h2 className="text-3xl font-bold mb-6">
              {language === 'en' ? 'Research Questions' : '研究问题'}
            </h2>
            <div className="grid gap-4">
              {researchQuestions.map((question, index) => (
                <div key={index} className="p-4 border border-gray-800 rounded-lg hover:border-purple-600/50 transition-colors bg-slate-800/20">
                  <p className="text-gray-300">{question}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Research Areas */}
          <section className="mb-12">
            <h2 className="text-3xl font-bold mb-6">
              {language === 'en' ? 'Research Areas' : '研究领域'}
            </h2>
            <div className="grid gap-6 md:grid-cols-2">
              {researchAreas.map((area, index) => (
                <div key={index} className="p-6 border border-gray-800 rounded-lg hover:border-purple-600/50 transition-colors bg-slate-800/20">
                  <h3 className="text-xl font-semibold mb-3 text-purple-400">{area.title}</h3>
                  <p className="text-gray-300">{area.description}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Audit Methodologies */}
          <section className="mb-12">
            <h2 className="text-3xl font-bold mb-6">
              {language === 'en' ? 'Audit Methodologies' : '审计方法'}
            </h2>
            <div className="grid gap-6 md:grid-cols-3">
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/20">
                <h3 className="text-lg font-semibold mb-3 text-pink-400">
                  {language === 'en' ? 'Content Analysis' : '内容分析'}
                </h3>
                <p className="text-gray-300 text-sm">
                  {language === 'en' 
                    ? "Systematic analysis of model responses to identify patterns of information suppression across different topics."
                    : "对模型响应的系统分析，以识别不同主题的信息抑制模式。"
                  }
                </p>
              </div>
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/20">
                <h3 className="text-lg font-semibold mb-3 text-pink-400">
                  {language === 'en' ? 'Semantic Detection' : '语义检测'}
                </h3>
                <p className="text-gray-300 text-sm">
                  {language === 'en' 
                    ? "Advanced techniques for detecting subtle forms of content filtering at the semantic level."
                    : "在语义层面检测微妙内容过滤形式的先进技术。"
                  }
                </p>
              </div>
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/20">
                <h3 className="text-lg font-semibold mb-3 text-pink-400">
                  {language === 'en' ? 'Quantitative Metrics' : '定量指标'}
                </h3>
                <p className="text-gray-300 text-sm">
                  {language === 'en' 
                    ? "Development of robust metrics to measure suppression rates and content accessibility patterns."
                    : "开发强大的指标来测量抑制率和内容可访问性模式。"
                  }
                </p>
              </div>
            </div>
          </section>

          {/* Paper Information */}
          <section className="mb-12">
            <h2 className="text-3xl font-bold mb-6">
              {language === 'en' ? 'Research Paper' : '研究论文'}
            </h2>
            <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/30">
              <h3 className="text-xl font-semibold mb-3">
                "Information Suppression in Large Language Models: Auditing, Quantifying, and Characterizing Censorship in DeepSeek"
              </h3>
              <p className="text-gray-400 mb-4">
                {language === 'en' 
                  ? "This research audits DeepSeek's LLMs to identify and quantify instances of information suppression, particularly concerning politically sensitive content."
                  : "这项研究审计了DeepSeek的大语言模型，以识别和量化信息抑制的实例，特别是涉及政治敏感内容的情况。"
                }
              </p>
              <div className="space-y-2">
                <a 
                  href="https://arxiv.org/abs/2506.12349" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="block text-purple-400 hover:text-purple-300 transition-colors"
                >
                  📄 View Paper on arXiv
                </a>
                <div className="text-sm text-gray-500">
                  {language === 'en' ? 'Published: June 2025' : '发布时间：2025年6月'}
                </div>
              </div>
            </div>
          </section>

          {/* Back Navigation */}
          <div className="text-center">
            <Link 
              href="/research/deepseek" 
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-slate-700/50 to-slate-600/50 border border-slate-500/50 rounded-lg hover:border-purple-500/50 hover:bg-slate-600/50 transition-all duration-200"
            >
              <span>←</span>
              <span>{language === 'en' ? 'Back to DeepSeek Research' : '返回DeepSeek研究'}</span>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
