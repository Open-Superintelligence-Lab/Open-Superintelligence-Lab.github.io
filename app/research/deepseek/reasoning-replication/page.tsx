'use client';

import Link from "next/link";
import { Navigation } from "@/components/navigation";
import { useLanguage } from "@/components/providers/language-provider";

export default function ReasoningReplicationPage() {
  const { language } = useLanguage();

  const researchQuestions = language === 'en' ? [
    "What data preparation and method designs are essential for effective SFT in replicating DeepSeek-R1?",
    "How can RLVR be implemented to enhance the reasoning capabilities of LLMs?",
    "What benchmarks are appropriate for evaluating the success of replication studies in LLMs?",
    "How do different training procedures affect the final reasoning performance of replicated models?",
    "What are the computational requirements and scaling factors for successful replication?",
    "How can we ensure reproducibility across different hardware configurations and environments?",
    "What are the key hyperparameters that significantly impact reasoning model performance?",
    "How do data quality and diversity affect the success of reasoning model replication?"
  ] : [
    "有效复制DeepSeek-R1的SFT需要什么数据准备和方法设计？",
    "如何实现RLVR以增强大语言模型的推理能力？",
    "评估大语言模型复制研究成功的适当基准是什么？",
    "不同的训练程序如何影响复制模型的最终推理性能？",
    "成功复制所需的计算要求和扩展因素是什么？",
    "我们如何确保在不同硬件配置和环境中的可重现性？",
    "显著影响推理模型性能的关键超参数是什么？",
    "数据质量和多样性如何影响推理模型复制的成功？"
  ];

  const researchAreas = language === 'en' ? [
    {
      title: "Supervised Fine-Tuning (SFT)",
      description: "Research optimal data preparation techniques and method designs for effective supervised fine-tuning in reasoning model replication."
    },
    {
      title: "Reinforcement Learning from Verifiable Rewards (RLVR)",
      description: "Develop and implement RLVR techniques to enhance reasoning capabilities and improve model performance on complex tasks."
    },
    {
      title: "Performance Benchmarking",
      description: "Create comprehensive benchmarks for evaluating reasoning model performance and replication success across different domains."
    },
    {
      title: "Training Optimization",
      description: "Optimize training procedures, hyperparameters, and scaling strategies for efficient reasoning model development."
    }
  ] : [
    {
      title: "监督微调(SFT)",
      description: "研究推理模型复制中有效监督微调的最佳数据准备技术和方法设计。"
    },
    {
      title: "可验证奖励强化学习(RLVR)",
      description: "开发和实施RLVR技术以增强推理能力并提高模型在复杂任务上的性能。"
    },
    {
      title: "性能基准测试",
      description: "创建全面的基准来评估推理模型性能和不同领域的复制成功。"
    },
    {
      title: "训练优化",
      description: "优化训练程序、超参数和扩展策略，以实现高效的推理模型开发。"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <Navigation />
      
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-600/20 via-blue-600/20 to-cyan-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-cyan-500/5 to-transparent"></div>
        </div>
        
        <div className="relative container mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
              Reasoning Model Replication
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              {language === 'en' 
                ? "100-day replication study of DeepSeek-R1 training procedures and reasoning capabilities"
                : "DeepSeek-R1训练程序和推理能力的100天复制研究"
              }
            </p>
            <div className="flex justify-center gap-4">
              <span className="bg-cyan-500/20 text-cyan-400 text-sm px-3 py-1 rounded-md">Replication Study</span>
              <span className="bg-blue-500/20 text-blue-400 text-sm px-3 py-1 rounded-md">Reasoning AI</span>
            </div>
          </div>
        </div>
      </section>

      <main className="container mx-auto px-6 py-12">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <Link 
              href="/research/deepseek-v3-2-exp" 
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
                  ? "This comprehensive survey reviews replication studies of DeepSeek-R1, focusing on training procedures and strategies to reproduce its performance. The research examines the 100-day replication journey, analyzing successful methodologies, identifying challenges, and providing insights for future reasoning model development."
                  : "这项综合调查研究了DeepSeek-R1的复制研究，专注于训练程序和重现其性能的策略。研究审视了100天的复制历程，分析了成功的方法论，识别了挑战，并为未来的推理模型开发提供了见解。"
                }
              </p>
              
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/30">
                <h3 className="text-xl font-semibold mb-3 text-cyan-400">
                  {language === 'en' ? 'Key Achievements' : '主要成就'}
                </h3>
                <ul className="space-y-2 text-gray-300">
                  <li>• {language === 'en' ? 'Successfully replicated DeepSeek-R1 reasoning capabilities' : '成功复制了DeepSeek-R1的推理能力'}</li>
                  <li>• {language === 'en' ? 'Developed optimized training procedures and methodologies' : '开发了优化的训练程序和方法论'}</li>
                  <li>• {language === 'en' ? 'Established comprehensive benchmarking frameworks' : '建立了全面的基准测试框架'}</li>
                  <li>• {language === 'en' ? 'Created reproducible training protocols for the community' : '为社区创建了可重现的训练协议'}</li>
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
                <div key={index} className="p-4 border border-gray-800 rounded-lg hover:border-cyan-600/50 transition-colors bg-slate-800/20">
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
                <div key={index} className="p-6 border border-gray-800 rounded-lg hover:border-cyan-600/50 transition-colors bg-slate-800/20">
                  <h3 className="text-xl font-semibold mb-3 text-cyan-400">{area.title}</h3>
                  <p className="text-gray-300">{area.description}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Training Components */}
          <section className="mb-12">
            <h2 className="text-3xl font-bold mb-6">
              {language === 'en' ? 'Training Components' : '训练组件'}
            </h2>
            <div className="grid gap-6 md:grid-cols-3">
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/20">
                <h3 className="text-lg font-semibold mb-3 text-blue-400">
                  {language === 'en' ? 'Data Preparation' : '数据准备'}
                </h3>
                <p className="text-gray-300 text-sm">
                  {language === 'en' 
                    ? "Systematic approach to preparing high-quality training data for reasoning model development."
                    : "为推理模型开发准备高质量训练数据的系统方法。"
                  }
                </p>
              </div>
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/20">
                <h3 className="text-lg font-semibold mb-3 text-blue-400">
                  {language === 'en' ? 'SFT Optimization' : 'SFT优化'}
                </h3>
                <p className="text-gray-300 text-sm">
                  {language === 'en' 
                    ? "Advanced supervised fine-tuning techniques optimized for reasoning task performance."
                    : "为推理任务性能优化的先进监督微调技术。"
                  }
                </p>
              </div>
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/20">
                <h3 className="text-lg font-semibold mb-3 text-blue-400">
                  {language === 'en' ? 'RLVR Implementation' : 'RLVR实现'}
                </h3>
                <p className="text-gray-300 text-sm">
                  {language === 'en' 
                    ? "Implementation of reinforcement learning from verifiable rewards for enhanced reasoning."
                    : "实施可验证奖励强化学习以增强推理能力。"
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
                "100 Days After DeepSeek-R1: A Survey on Replication Studies and More Directions for Reasoning Language Models"
              </h3>
              <p className="text-gray-400 mb-4">
                {language === 'en' 
                  ? "This survey reviews replication studies of DeepSeek-R1, focusing on training procedures and strategies to reproduce its performance."
                  : "这项调查研究了DeepSeek-R1的复制研究，专注于训练程序和重现其性能的策略。"
                }
              </p>
              <div className="space-y-2">
                <a 
                  href="https://arxiv.org/abs/2505.00551" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="block text-cyan-400 hover:text-cyan-300 transition-colors"
                >
                  📄 View Paper on arXiv
                </a>
                <div className="text-sm text-gray-500">
                  {language === 'en' ? 'Published: May 2025' : '发布时间：2025年5月'}
                </div>
              </div>
            </div>
          </section>

          {/* Back Navigation */}
          <div className="text-center">
            <Link 
              href="/research/deepseek-v3-2-exp" 
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-slate-700/50 to-slate-600/50 border border-slate-500/50 rounded-lg hover:border-cyan-500/50 hover:bg-slate-600/50 transition-all duration-200"
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
