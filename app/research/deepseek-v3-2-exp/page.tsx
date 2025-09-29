'use client';

import Link from "next/link";
import { Navigation } from "@/components/navigation";
import { useLanguage } from "@/components/providers/language-provider";
import { translations } from "@/lib/language-detection";

export default function DeepSeekResearchPage() {
  const { language } = useLanguage();
  const t = translations[language];

  const researchQuestions = language === 'en' ? [
    "How can we optimize the lightning indexer for even better computational efficiency?",
    "What are the optimal sparse attention patterns for different types of long-context tasks?",
    "How does DSA perform on multimodal tasks with long sequences?",
    "Can we develop adaptive sparse attention that adjusts k based on context complexity?",
    "What are the theoretical limits of sparse attention while maintaining model performance?",
    "How can we improve the fine-grained token selection mechanism?",
    "What impact does DSA have on different model architectures beyond MLA?",
    "How can we optimize DSA for edge devices and mobile deployment?"
  ] : [
    "我们如何优化闪电索引器以获得更好的计算效率？",
    "对于不同类型的长上下文任务，最优的稀疏注意力模式是什么？",
    "DSA在多模态长序列任务上的表现如何？",
    "我们能否开发自适应稀疏注意力，根据上下文复杂性调整k值？",
    "在保持模型性能的同时，稀疏注意力的理论极限是什么？",
    "我们如何改进细粒度token选择机制？",
    "DSA对MLA之外的不同模型架构有什么影响？",
    "我们如何优化DSA以用于边缘设备和移动部署？"
  ];

  const contributionAreas = language === 'en' ? [
    {
      title: "Implementation & Optimization",
      description: "Contribute to the open-source implementation, optimize CUDA kernels, and improve inference speed."
    },
    {
      title: "Research & Analysis", 
      description: "Analyze sparse attention patterns, conduct ablation studies, and publish research findings."
    },
    {
      title: "Benchmarking & Evaluation",
      description: "Develop new benchmarks for long-context tasks and evaluate DSA performance across domains."
    },
    {
      title: "Documentation & Tutorials",
      description: "Create comprehensive documentation, tutorials, and educational content for the community."
    }
  ] : [
    {
      title: "实现与优化",
      description: "为开源实现做出贡献，优化CUDA内核，提高推理速度。"
    },
    {
      title: "研究与分析",
      description: "分析稀疏注意力模式，进行消融研究，发表研究成果。"
    },
    {
      title: "基准测试与评估",
      description: "为长上下文任务开发新的基准测试，评估DSA在各领域的性能。"
    },
    {
      title: "文档与教程",
      description: "为社区创建全面的文档、教程和教育内容。"
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
            {t.deepseekTitle}
          </h1>
          
          <div className="prose prose-invert max-w-none">
            <p className="text-xl text-gray-300 mb-8">
              {t.deepseekDescription}
            </p>
            
            <div className="space-y-12">
              {/* Research Path */}
              <section>
                <h2 className="text-3xl font-semibold mb-6">{t.deepseekResearchPath}</h2>
                <div className="space-y-4">
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">1. Architecture Analysis</h3>
                    <p className="text-gray-300">
                      {language === 'en' 
                        ? "Study the DeepSeek Sparse Attention (DSA) architecture, including the lightning indexer and fine-grained token selection mechanism."
                        : "研究DeepSeek稀疏注意力(DSA)架构，包括闪电索引器和细粒度token选择机制。"
                      }
                    </p>
                  </div>
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">2. Implementation Research</h3>
                    <p className="text-gray-300">
                      {language === 'en'
                        ? "Explore the open-source implementation, understand the MQA mode of MLA, and identify optimization opportunities."
                        : "探索开源实现，理解MLA的MQA模式，识别优化机会。"
                      }
                    </p>
                  </div>
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">3. Performance Evaluation</h3>
                    <p className="text-gray-300">
                      {language === 'en'
                        ? "Benchmark DSA performance across different tasks, analyze efficiency gains, and identify limitations."
                        : "在不同任务上对DSA性能进行基准测试，分析效率提升，识别局限性。"
                      }
                    </p>
                  </div>
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">4. Innovation & Extension</h3>
                    <p className="text-gray-300">
                      {language === 'en'
                        ? "Develop novel improvements, explore new applications, and contribute to the open-source ecosystem."
                        : "开发新颖的改进，探索新的应用，为开源生态系统做出贡献。"
                      }
                    </p>
                  </div>
                </div>
              </section>

              {/* Research Questions */}
              <section>
                <h2 className="text-3xl font-semibold mb-6">{t.deepseekQuestions}</h2>
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
                <h2 className="text-3xl font-semibold mb-6">{t.deepseekContributions}</h2>
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
                <h2 className="text-3xl font-semibold mb-6">{t.deepseekOpenSource}</h2>
                <div className="space-y-6">
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">
                      {language === 'en' ? 'Repository & Resources' : '仓库与资源'}
                    </h3>
                    <div className="space-y-3">
                      <a 
                        href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="block text-blue-400 hover:text-blue-300 transition-colors"
                      >
                        🤗 Hugging Face Model Repository
                      </a>
                      <a 
                        href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="block text-blue-400 hover:text-blue-300 transition-colors"
                      >
                        📁 Open Source Implementation
                      </a>
                    </div>
                  </div>
                  
                  <div className="p-6 border border-gray-800 rounded-lg">
                    <h3 className="text-xl font-semibold mb-3">
                      {language === 'en' ? 'Key Research Areas' : '关键研究领域'}
                    </h3>
                    <ul className="space-y-2 text-gray-300">
                      <li>• {language === 'en' ? 'Sparse Attention Mechanisms' : '稀疏注意力机制'}</li>
                      <li>• {language === 'en' ? 'Long-Context Efficiency' : '长上下文效率'}</li>
                      <li>• {language === 'en' ? 'Lightning Indexer Optimization' : '闪电索引器优化'}</li>
                      <li>• {language === 'en' ? 'Multi-Query Attention (MQA)' : '多查询注意力(MQA)'}</li>
                      <li>• {language === 'en' ? 'Mixture-of-Latent-Attention (MLA)' : '混合潜在注意力(MLA)'}</li>
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
