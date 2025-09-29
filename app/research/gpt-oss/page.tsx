'use client';

import Link from "next/link";
import { Navigation } from "@/components/navigation";
import { useLanguage } from "@/components/providers/language-provider";
import { translations } from "@/lib/language-detection";

export default function GPTOSSResearchPage() {
  const { language } = useLanguage();
  const t = translations[language];

  const researchQuestions = language === 'en' ? [
    "How can we improve multilingual performance in GPT-OSS models, especially for non-English languages?",
    "What are the optimal MoE routing strategies for different types of reasoning tasks?",
    "How can we enhance the OpenAI Harmony response format for better safety and alignment?",
    "What are the best practices for deploying GPT-OSS models on edge devices with limited resources?",
    "How can we optimize the MXFP4 quantization technique for better memory efficiency?",
    "What improvements can be made to the chain-of-thought monitoring system?",
    "How does GPT-OSS perform on specialized domains compared to other open-source models?",
    "What are the security implications of running GPT-OSS models locally vs. cloud deployment?"
  ] : [
    "我们如何改善GPT-OSS模型的多语言性能，特别是非英语语言？",
    "对于不同类型的推理任务，最优的MoE路由策略是什么？",
    "我们如何增强OpenAI Harmony响应格式以获得更好的安全性和对齐？",
    "在资源有限的边缘设备上部署GPT-OSS模型的最佳实践是什么？",
    "我们如何优化MXFP4量化技术以获得更好的内存效率？",
    "我们可以对思维链监控系统做哪些改进？",
    "GPT-OSS在专业领域与其他开源模型相比表现如何？",
    "本地运行GPT-OSS模型与云部署的安全影响是什么？"
  ];

  const contributionAreas = language === 'en' ? [
    {
      title: "Model Optimization",
      description: "Improve MoE routing efficiency, optimize quantization techniques, and enhance edge deployment capabilities."
    },
    {
      title: "Multilingual Enhancement", 
      description: "Develop better multilingual support, improve non-English language performance, and create language-specific fine-tuning strategies."
    },
    {
      title: "Safety & Alignment",
      description: "Enhance OpenAI Harmony format, improve chain-of-thought monitoring, and develop better safety mechanisms."
    },
    {
      title: "Applications & Tools",
      description: "Build applications leveraging GPT-OSS capabilities, create deployment tools, and develop evaluation frameworks."
    }
  ] : [
    {
      title: "模型优化",
      description: "改善MoE路由效率，优化量化技术，增强边缘部署能力。"
    },
    {
      title: "多语言增强",
      description: "开发更好的多语言支持，改善非英语语言性能，创建特定语言的微调策略。"
    },
    {
      title: "安全与对齐",
      description: "增强OpenAI Harmony格式，改善思维链监控，开发更好的安全机制。"
    },
    {
      title: "应用与工具",
      description: "构建利用GPT-OSS能力的应用程序，创建部署工具，开发评估框架。"
    }
  ];

  const modelSpecs = language === 'en' ? [
    {
      model: "GPT-OSS-120B",
      parameters: "117B total, 5.1B active per token",
      experts: "128 experts, 4 active routing",
      context: "128K context window",
      quantization: "MXFP4 quantization"
    },
    {
      model: "GPT-OSS-20B", 
      parameters: "21B total, 3.6B active per token",
      experts: "32 experts, 4 active routing",
      context: "Optimized for edge deployment",
      quantization: "MXFP4 quantization"
    }
  ] : [
    {
      model: "GPT-OSS-120B",
      parameters: "1170亿总参数，每token激活51亿参数",
      experts: "128个专家，4个活跃路由",
      context: "128K上下文窗口",
      quantization: "MXFP4量化"
    },
    {
      model: "GPT-OSS-20B",
      parameters: "210亿总参数，每token激活36亿参数", 
      experts: "32个专家，4个活跃路由",
      context: "针对边缘部署优化",
      quantization: "MXFP4量化"
    }
  ];

  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation />
      
      <main className="container mx-auto px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <Link 
              href="/" 
              className="text-gray-400 hover:text-white transition-colors"
            >
              ← Back to Home
            </Link>
          </div>
          
          <h1 className="text-5xl font-bold mb-6">
            {t.gptOssTitle}
          </h1>
          
          <div className="prose prose-invert max-w-none">
            <p className="text-xl text-gray-300 mb-8">
              {t.gptOssDescription}
            </p>
            
            <div className="space-y-12">
              {/* Model Specifications */}
              <section>
                <h2 className="text-3xl font-semibold mb-6">
                  {language === 'en' ? 'Model Specifications' : '模型规格'}
                </h2>
                <div className="grid gap-6 md:grid-cols-2">
                  {modelSpecs.map((spec, index) => (
                    <div key={index} className="p-6 border border-gray-800 rounded-lg">
                      <h3 className="text-xl font-semibold mb-4">{spec.model}</h3>
                      <div className="space-y-2 text-gray-300">
                        <p><strong>{language === 'en' ? 'Parameters:' : '参数:'}</strong> {spec.parameters}</p>
                        <p><strong>{language === 'en' ? 'Experts:' : '专家:'}</strong> {spec.experts}</p>
                        <p><strong>{language === 'en' ? 'Context:' : '上下文:'}</strong> {spec.context}</p>
                        <p><strong>{language === 'en' ? 'Quantization:' : '量化:'}</strong> {spec.quantization}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* Research Path */}
              <section>
                <h2 className="text-3xl font-semibold mb-6">{t.gptOssResearchPath}</h2>
                <div className="space-y-4">
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">1. Architecture Analysis</h3>
                    <p className="text-gray-300">
                      {language === 'en' 
                        ? "Study the Mixture-of-Experts (MoE) architecture, understand routing mechanisms, and analyze computational efficiency."
                        : "研究混合专家(MoE)架构，理解路由机制，分析计算效率。"
                      }
                    </p>
                  </div>
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">2. Safety & Alignment Research</h3>
                    <p className="text-gray-300">
                      {language === 'en'
                        ? "Investigate OpenAI Harmony response format, analyze chain-of-thought monitoring, and explore safety mechanisms."
                        : "研究OpenAI Harmony响应格式，分析思维链监控，探索安全机制。"
                      }
                    </p>
                  </div>
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">3. Performance Optimization</h3>
                    <p className="text-gray-300">
                      {language === 'en'
                        ? "Optimize MXFP4 quantization, improve multilingual performance, and enhance edge deployment capabilities."
                        : "优化MXFP4量化，改善多语言性能，增强边缘部署能力。"
                      }
                    </p>
                  </div>
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">4. Application Development</h3>
                    <p className="text-gray-300">
                      {language === 'en'
                        ? "Build applications, create deployment tools, and develop evaluation frameworks for GPT-OSS models."
                        : "构建应用程序，创建部署工具，为GPT-OSS模型开发评估框架。"
                      }
                    </p>
                  </div>
                </div>
              </section>

              {/* Research Questions */}
              <section>
                <h2 className="text-3xl font-semibold mb-6">{t.gptOssQuestions}</h2>
                <div className="grid gap-4">
                  {researchQuestions.map((question, index) => (
                    <div key={index} className="p-4 border border-gray-800 rounded-lg hover:border-gray-600 transition-colors">
                      <p className="text-gray-300">{question}</p>
                    </div>
                  ))}
                </div>
              </section>

              {/* How to Contribute */}
              <section>
                <h2 className="text-3xl font-semibold mb-6">{t.gptOssContributions}</h2>
                <div className="grid gap-6 md:grid-cols-2">
                  {contributionAreas.map((area, index) => (
                    <div key={index} className="p-6 border border-gray-800 rounded-lg hover:border-gray-600 transition-colors">
                      <h3 className="text-xl font-semibold mb-3">{area.title}</h3>
                      <p className="text-gray-300">{area.description}</p>
                    </div>
                  ))}
                </div>
              </section>

              {/* Open Source Research */}
              <section>
                <h2 className="text-3xl font-semibold mb-6">{t.gptOssOpenSource}</h2>
                <div className="space-y-6">
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">
                      {language === 'en' ? 'Repository & Resources' : '仓库与资源'}
                    </h3>
                    <div className="space-y-3">
                      <a 
                        href="https://github.com/openai/gpt-oss" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="block text-blue-400 hover:text-blue-300 transition-colors"
                      >
                        🐙 GitHub Repository
                      </a>
                      <a 
                        href="https://openai.com/blog/introducing-gpt-oss/" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="block text-blue-400 hover:text-blue-300 transition-colors"
                      >
                        📖 Official Blog Post
                      </a>
                    </div>
                  </div>
                  
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">
                      {language === 'en' ? 'Key Research Areas' : '关键研究领域'}
                    </h3>
                    <ul className="space-y-2 text-gray-300">
                      <li>• {language === 'en' ? 'Mixture-of-Experts (MoE) Architecture' : '混合专家(MoE)架构'}</li>
                      <li>• {language === 'en' ? 'OpenAI Harmony Response Format' : 'OpenAI Harmony响应格式'}</li>
                      <li>• {language === 'en' ? 'MXFP4 Quantization' : 'MXFP4量化'}</li>
                      <li>• {language === 'en' ? 'Chain-of-Thought Monitoring' : '思维链监控'}</li>
                      <li>• {language === 'en' ? 'Edge Deployment Optimization' : '边缘部署优化'}</li>
                      <li>• {language === 'en' ? 'Multilingual Performance' : '多语言性能'}</li>
                    </ul>
                  </div>
                </div>
              </section>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
