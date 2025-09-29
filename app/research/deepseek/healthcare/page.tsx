'use client';

import Link from "next/link";
import { Navigation } from "@/components/navigation";
import { useLanguage } from "@/components/providers/language-provider";

export default function HealthcarePage() {
  const { language } = useLanguage();

  const researchQuestions = language === 'en' ? [
    "What techniques are effective for fine-tuning DeepSeek-R1 to enhance performance in healthcare diagnostics?",
    "How can biases in LLMs be identified and mitigated, especially in sensitive domains like healthcare?",
    "What methodologies are suitable for validating the accuracy and reliability of LLMs in clinical decision support tasks?",
    "How does DeepSeek-R1 perform on specialized medical reasoning tasks compared to domain-specific models?",
    "What are the ethical implications of using LLMs in healthcare decision-making?",
    "How can we ensure patient privacy and data security when using DeepSeek-R1 in clinical settings?",
    "What regulatory considerations apply to deploying LLMs in healthcare applications?",
    "How can we measure and improve the interpretability of LLM decisions in medical contexts?"
  ] : [
    "哪些技术对于微调DeepSeek-R1以提升医疗诊断性能是有效的？",
    "如何识别和减轻大语言模型中的偏见，特别是在医疗等敏感领域？",
    "哪些方法适合验证大语言模型在临床决策支持任务中的准确性和可靠性？",
    "DeepSeek-R1在专业医疗推理任务上的表现与领域特定模型相比如何？",
    "在医疗决策中使用大语言模型的伦理影响是什么？",
    "在临床环境中使用DeepSeek-R1时，如何确保患者隐私和数据安全？",
    "在医疗应用中部署大语言模型需要考虑哪些监管问题？",
    "如何测量和改进大语言模型在医疗背景下的决策可解释性？"
  ];

  const researchAreas = language === 'en' ? [
    {
      title: "Clinical Decision Support",
      description: "Develop and evaluate DeepSeek-R1 applications for clinical decision support, diagnostic assistance, and treatment recommendations."
    },
    {
      title: "Bias Mitigation",
      description: "Identify and develop methods to mitigate biases in LLMs, particularly in healthcare contexts where bias can have serious consequences."
    },
    {
      title: "Model Fine-tuning",
      description: "Explore effective fine-tuning techniques to adapt DeepSeek-R1 for specialized healthcare tasks and medical reasoning."
    },
    {
      title: "Validation & Safety",
      description: "Develop robust validation methodologies and safety protocols for deploying LLMs in healthcare environments."
    }
  ] : [
    {
      title: "临床决策支持",
      description: "开发和评估DeepSeek-R1在临床决策支持、诊断辅助和治疗建议中的应用。"
    },
    {
      title: "偏见缓解",
      description: "识别和开发缓解大语言模型中偏见的方法，特别是在偏见可能产生严重后果的医疗背景下。"
    },
    {
      title: "模型微调",
      description: "探索有效的微调技术，使DeepSeek-R1适应专业医疗任务和医疗推理。"
    },
    {
      title: "验证与安全",
      description: "开发在医疗环境中部署大语言模型的强大验证方法和安全协议。"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <Navigation />
      
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-cyan-600/20 to-blue-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
        </div>
        
        <div className="relative container mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Healthcare AI Applications
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              {language === 'en' 
                ? "Survey of DeepSeek-R1 capabilities, risks, and clinical applications in healthcare diagnostics"
                : "DeepSeek-R1在医疗诊断中的能力、风险和临床应用调查"
              }
            </p>
            <div className="flex justify-center gap-4">
              <span className="bg-blue-500/20 text-blue-400 text-sm px-3 py-1 rounded-md">Healthcare AI</span>
              <span className="bg-cyan-500/20 text-cyan-400 text-sm px-3 py-1 rounded-md">Clinical Research</span>
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
                  ? "This comprehensive survey examines the application of DeepSeek-R1 in healthcare, assessing its reasoning capabilities, potential risks, and clinical applications. The research evaluates the model's performance on medical tasks, identifies areas for improvement, and discusses the ethical and practical considerations of deploying LLMs in healthcare settings."
                  : "这项综合调查研究了DeepSeek-R1在医疗中的应用，评估了其推理能力、潜在风险和临床应用。研究评估了模型在医疗任务上的表现，识别了改进领域，并讨论了在医疗环境中部署大语言模型的伦理和实际考虑。"
                }
              </p>
              
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/30">
                <h3 className="text-xl font-semibold mb-3 text-blue-400">
                  {language === 'en' ? 'Key Capabilities' : '主要能力'}
                </h3>
                <ul className="space-y-2 text-gray-300">
                  <li>• {language === 'en' ? 'Advanced medical reasoning and diagnostic assistance' : '先进的医疗推理和诊断辅助'}</li>
                  <li>• {language === 'en' ? 'Clinical decision support system integration' : '临床决策支持系统集成'}</li>
                  <li>• {language === 'en' ? 'Medical literature analysis and synthesis' : '医学文献分析和综合'}</li>
                  <li>• {language === 'en' ? 'Patient data interpretation and risk assessment' : '患者数据解释和风险评估'}</li>
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
                <div key={index} className="p-4 border border-gray-800 rounded-lg hover:border-blue-600/50 transition-colors bg-slate-800/20">
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
                <div key={index} className="p-6 border border-gray-800 rounded-lg hover:border-blue-600/50 transition-colors bg-slate-800/20">
                  <h3 className="text-xl font-semibold mb-3 text-blue-400">{area.title}</h3>
                  <p className="text-gray-300">{area.description}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Clinical Applications */}
          <section className="mb-12">
            <h2 className="text-3xl font-bold mb-6">
              {language === 'en' ? 'Clinical Applications' : '临床应用'}
            </h2>
            <div className="grid gap-6 md:grid-cols-3">
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/20">
                <h3 className="text-lg font-semibold mb-3 text-cyan-400">
                  {language === 'en' ? 'Diagnostic Support' : '诊断支持'}
                </h3>
                <p className="text-gray-300 text-sm">
                  {language === 'en' 
                    ? "Assist healthcare professionals in diagnostic processes through advanced reasoning and medical knowledge synthesis."
                    : "通过先进的推理和医学知识综合协助医疗专业人员进行诊断过程。"
                  }
                </p>
              </div>
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/20">
                <h3 className="text-lg font-semibold mb-3 text-cyan-400">
                  {language === 'en' ? 'Treatment Planning' : '治疗计划'}
                </h3>
                <p className="text-gray-300 text-sm">
                  {language === 'en' 
                    ? "Generate evidence-based treatment recommendations and care plans based on patient data and medical guidelines."
                    : "基于患者数据和医疗指南生成循证治疗建议和护理计划。"
                  }
                </p>
              </div>
              <div className="p-6 border border-gray-800 rounded-lg bg-slate-800/20">
                <h3 className="text-lg font-semibold mb-3 text-cyan-400">
                  {language === 'en' ? 'Risk Assessment' : '风险评估'}
                </h3>
                <p className="text-gray-300 text-sm">
                  {language === 'en' 
                    ? "Analyze patient risk factors and predict potential health outcomes to support preventive care strategies."
                    : "分析患者风险因素并预测潜在健康结果，以支持预防性护理策略。"
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
                "DeepSeek in Healthcare: A Survey of Capabilities, Risks, and Clinical Applications of Open-Source Large Language Models"
              </h3>
              <p className="text-gray-400 mb-4">
                {language === 'en' 
                  ? "This survey examines the application of DeepSeek-R1 in healthcare, assessing its reasoning capabilities, potential risks, and clinical applications."
                  : "这项调查研究了DeepSeek-R1在医疗中的应用，评估了其推理能力、潜在风险和临床应用。"
                }
              </p>
              <div className="space-y-2">
                <a 
                  href="https://arxiv.org/abs/2506.01257" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="block text-blue-400 hover:text-blue-300 transition-colors"
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
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-slate-700/50 to-slate-600/50 border border-slate-500/50 rounded-lg hover:border-blue-500/50 hover:bg-slate-600/50 transition-all duration-200"
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
