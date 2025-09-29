'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function VisualHallucinationsPage() {
  const { language } = useLanguage();

  const researchQuestions = language === 'en' ? [
    "How can image embeddings be systematically optimized to induce specific visual hallucinations in MLLMs?",
    "What methods can be employed to adapt embedding manipulation attacks for models like DeepSeek Janus?",
    "How can hallucination rates and visual fidelity be quantitatively assessed in manipulated images?",
    "What are the underlying representation vulnerabilities that make MLLMs susceptible to such attacks?",
    "How can we develop robust defenses against embedding manipulation attacks?",
    "What impact do different image encoders have on the success of hallucination attacks?",
    "Can we predict which types of images are most vulnerable to targeted hallucinations?",
    "How do hallucination patterns vary across different multimodal model architectures?"
  ] : [
    "如何系统性地优化图像嵌入以在多模态大语言模型中诱导特定的视觉幻觉？",
    "可以采用什么方法来使嵌入操作攻击适应DeepSeek Janus等模型？",
    "如何在被操纵的图像中定量评估幻觉率和视觉保真度？",
    "使多模态大语言模型容易受到此类攻击的潜在表示漏洞是什么？",
    "我们如何开发针对嵌入操作攻击的强大防御？",
    "不同的图像编码器对幻觉攻击成功的影响是什么？",
    "我们能否预测哪些类型的图像最容易受到目标幻觉的影响？",
    "幻觉模式在不同多模态模型架构中如何变化？"
  ];

  const researchAreas = language === 'en' ? [
    {
      title: "Embedding Manipulation Techniques",
      description: "Develop and analyze methods for systematically modifying image embeddings to induce targeted visual hallucinations."
    },
    {
      title: "Attack Adaptation",
      description: "Adapt existing embedding manipulation attacks for different multimodal model architectures like DeepSeek Janus."
    },
    {
      title: "Evaluation Metrics",
      description: "Design quantitative metrics for assessing hallucination rates, visual fidelity, and attack success rates."
    },
    {
      title: "Vulnerability Analysis",
      description: "Analyze the underlying representation vulnerabilities that make MLLMs susceptible to manipulation attacks."
    }
  ] : [
    {
      title: "嵌入操作技术",
      description: "开发和分析系统性地修改图像嵌入以诱导目标视觉幻觉的方法。"
    },
    {
      title: "攻击适应",
      description: "使现有的嵌入操作攻击适应不同的多模态模型架构，如DeepSeek Janus。"
    },
    {
      title: "评估指标",
      description: "设计定量指标来评估幻觉率、视觉保真度和攻击成功率。"
    },
    {
      title: "漏洞分析",
      description: "分析使多模态大语言模型容易受到操作攻击的潜在表示漏洞。"
    }
  ];

  return (
    <>
      
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-red-600/20 via-orange-600/20 to-red-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-red-500/5 to-transparent"></div>
        </div>
        
        <div className="relative container mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text text-transparent">
              Visual Hallucination Vulnerabilities
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              {language === 'en' 
                ? "Research on inducing targeted visual hallucinations in multimodal LLMs through embedding manipulation attacks"
                : "通过嵌入操作攻击在多模态大语言模型中诱导目标视觉幻觉的研究"
              }
            </p>
            <div className="flex justify-center gap-4">
              <span className="bg-red-500/20 text-red-400 text-sm px-3 py-1 rounded-md">Security Research</span>
              <span className="bg-orange-500/20 text-orange-400 text-sm px-3 py-1 rounded-md">Multimodal AI</span>
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
                  ? "This research explores vulnerabilities in DeepSeek's multimodal large language models (MLLMs), focusing on inducing targeted visual hallucinations through embedding manipulation attacks. The study investigates how image embeddings can be systematically optimized to produce specific visual hallucinations in models like DeepSeek Janus."
                  : "这项研究探索了DeepSeek多模态大语言模型(MLLMs)中的漏洞，专注于通过嵌入操作攻击诱导目标视觉幻觉。研究调查了如何系统性地优化图像嵌入以在DeepSeek Janus等模型中产生特定的视觉幻觉。"
                }
              </p>
              
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/30">
                <h3 className="text-xl font-semibold mb-3 text-red-400">
                  {language === 'en' ? 'Key Findings' : '主要发现'}
                </h3>
                <ul className="space-y-2 text-gray-300">
                  <li>• {language === 'en' ? 'Identified systematic vulnerabilities in multimodal model representations' : '识别了多模态模型表示中的系统性漏洞'}</li>
                  <li>• {language === 'en' ? 'Developed effective embedding manipulation techniques for targeted hallucinations' : '开发了针对目标幻觉的有效嵌入操作技术'}</li>
                  <li>• {language === 'en' ? 'Quantified hallucination rates and visual fidelity in attacked models' : '量化了被攻击模型中的幻觉率和视觉保真度'}</li>
                  <li>• {language === 'en' ? 'Demonstrated cross-model transferability of attack methods' : '证明了攻击方法的跨模型可转移性'}</li>
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
                <div key={index} className="p-4 border border-gray-800 rounded-lg hover:border-red-600/50 transition-colors bg-slate-800/20">
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
                <div key={index} className="p-6 border border-gray-800 rounded-lg hover:border-red-600/50 transition-colors bg-slate-800/20">
                  <h3 className="text-xl font-semibold mb-3 text-red-400">{area.title}</h3>
                  <p className="text-gray-300">{area.description}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Paper Information */}
          <section className="mb-12">
            <h2 className="text-3xl font-bold mb-6">
              {language === 'en' ? 'Research Paper' : '研究论文'}
            </h2>
            <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/30">
              <h3 className="text-xl font-semibold mb-3">
                "DeepSeek on a Trip: Inducing Targeted Visual Hallucinations via Representation Vulnerabilities"
              </h3>
              <p className="text-gray-400 mb-4">
                {language === 'en' 
                  ? "This study explores vulnerabilities in DeepSeek's multimodal large language models (MLLMs), focusing on inducing targeted visual hallucinations through embedding manipulation attacks."
                  : "这项研究探索了DeepSeek多模态大语言模型(MLLMs)中的漏洞，专注于通过嵌入操作攻击诱导目标视觉幻觉。"
                }
              </p>
              <div className="space-y-2">
                <a 
                  href="https://arxiv.org/abs/2502.07905" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="block text-red-400 hover:text-red-300 transition-colors"
                >
                  📄 View Paper on arXiv
                </a>
                <div className="text-sm text-gray-500">
                  {language === 'en' ? 'Published: February 2025' : '发布时间：2025年2月'}
                </div>
              </div>
            </div>
          </section>

          {/* Back Navigation */}
          <div className="text-center">
            <Link 
              href="/research/deepseek" 
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-slate-700/50 to-slate-600/50 border border-slate-500/50 rounded-lg hover:border-red-500/50 hover:bg-slate-600/50 transition-all duration-200"
            >
              <span>←</span>
              <span>{language === 'en' ? 'Back to DeepSeek Research' : '返回DeepSeek研究'}</span>
            </Link>
          </div>
        </div>
      </main>
    </>
  );
}
