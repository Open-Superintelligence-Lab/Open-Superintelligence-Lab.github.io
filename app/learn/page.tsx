'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function LearnPage() {
  const { language } = useLanguage();

  const dailyTasks = language === 'en' ? [
    {
      title: "Literature Review & Research",
      description: "Analyze latest papers, track SOTA models, and identify research gaps in cutting-edge AI developments.",
      icon: "📚"
    },
    {
      title: "Model Experimentation",
      description: "Design, implement, and test novel architectures, training methods, and optimization techniques.",
      icon: "🧪"
    },
    {
      title: "Data Analysis & Preprocessing",
      description: "Curate datasets, perform statistical analysis, and develop robust data pipelines for model training.",
      icon: "📊"
    },
    {
      title: "Code Implementation",
      description: "Write production-quality code, optimize algorithms, and contribute to open-source AI frameworks.",
      icon: "💻"
    },
    {
      title: "Benchmarking & Evaluation",
      description: "Design experiments, run comprehensive benchmarks, and analyze model performance across domains.",
      icon: "📈"
    },
    {
      title: "Collaboration & Communication",
      description: "Present findings, write technical reports, and collaborate with interdisciplinary research teams.",
      icon: "🤝"
    }
  ] : [
    {
      title: "文献综述与研究",
      description: "分析最新论文，跟踪SOTA模型，识别前沿AI发展中的研究空白。",
      icon: "📚"
    },
    {
      title: "模型实验",
      description: "设计、实施和测试新颖的架构、训练方法和优化技术。",
      icon: "🧪"
    },
    {
      title: "数据分析与预处理",
      description: "策划数据集，进行统计分析，开发用于模型训练的强大数据管道。",
      icon: "📊"
    },
    {
      title: "代码实现",
      description: "编写生产质量代码，优化算法，为开源AI框架做出贡献。",
      icon: "💻"
    },
    {
      title: "基准测试与评估",
      description: "设计实验，运行综合基准测试，分析跨领域模型性能。",
      icon: "📈"
    },
    {
      title: "协作与沟通",
      description: "展示发现，撰写技术报告，与跨学科研究团队合作。",
      icon: "🤝"
    }
  ];

  const skoolFeatures = language === 'en' ? [
    {
      title: "Daily Micro-Learning",
      description: "20 minutes of focused content every day, designed for busy schedules without overwhelming you.",
      icon: "⏰"
    },
    {
      title: "Exclusive Research Content",
      description: "Access to cutting-edge AI research insights, paper breakdowns, and industry trends before they go mainstream.",
      icon: "🔬"
    },
    {
      title: "Personal Guidance by Vuk",
      description: "Direct mentorship and feedback from an experienced AI researcher who's worked at top labs.",
      icon: "👨‍🔬"
    },
    {
      title: "Community Support",
      description: "Connect with like-minded researchers, share projects, and get feedback from a supportive community.",
      icon: "🌐"
    },
    {
      title: "Real Lab Experience",
      description: "Work on actual research projects and contribute to top-tier AI research from day one.",
      icon: "🏛️"
    },
    {
      title: "Career Acceleration",
      description: "From beginner to contributing AI researcher in just one year with structured progression.",
      icon: "🚀"
    }
  ] : [
    {
      title: "每日微学习",
      description: "每天20分钟的专注内容，专为忙碌的日程设计，不会让你感到压力。",
      icon: "⏰"
    },
    {
      title: "独家研究内容",
      description: "获取前沿AI研究见解、论文解析和行业趋势，比主流更早了解。",
      icon: "🔬"
    },
    {
      title: "Vuk的个人指导",
      description: "来自在顶级实验室工作过的经验丰富的AI研究员的直接指导和反馈。",
      icon: "👨‍🔬"
    },
    {
      title: "社区支持",
      description: "与志同道合的研究员联系，分享项目，并获得支持性社区的反馈。",
      icon: "🌐"
    },
    {
      title: "真实实验室体验",
      description: "从事实际研究项目，从第一天起就为顶级AI研究做出贡献。",
      icon: "🏛️"
    },
    {
      title: "职业加速",
      description: "通过结构化进步，在短短一年内从初学者成为贡献的AI研究员。",
      icon: "🚀"
    }
  ];

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-cyan-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
        </div>
        
        <div className="relative container mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
              {language === 'en' ? 'Become an AI Researcher' : '成为AI研究员'}
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              {language === 'en' 
                ? "Learn from scratch and contribute to cutting-edge AI research at top laboratories"
                : "从零开始学习，为顶级实验室的前沿AI研究做出贡献"
              }
            </p>
            <div className="flex justify-center gap-4 mb-8">
              <span className="bg-blue-500/20 text-blue-400 text-sm px-3 py-1 rounded-md">Research Training</span>
              <span className="bg-purple-500/20 text-purple-400 text-sm px-3 py-1 rounded-md">Daily Learning</span>
              <span className="bg-cyan-500/20 text-cyan-400 text-sm px-3 py-1 rounded-md">Community Support</span>
            </div>
          </div>
        </div>
      </section>

      <main className="container mx-auto px-6 py-12">
        {/* What AI Researchers Do Daily */}
        <section className="mb-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">
              {language === 'en' ? 'What AI Researchers Do Daily' : 'AI研究员日常工作'}
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              {language === 'en' 
                ? "Experience the real day-to-day activities of AI researchers working at top laboratories"
                : "体验在顶级实验室工作的AI研究员的真实日常活动"
              }
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
            {dailyTasks.map((task, index) => (
              <div key={index} className="p-6 border border-gray-800 rounded-xl bg-slate-800/30 hover:border-blue-500/50 transition-all duration-300">
                <div className="text-3xl mb-4">{task.icon}</div>
                <h3 className="text-xl font-semibold mb-3 text-blue-400">{task.title}</h3>
                <p className="text-gray-300 text-sm leading-relaxed">{task.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Skool Program Section */}
        <section className="mb-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">
              {language === 'en' ? '20 Minutes a Day Program' : '每日20分钟计划'}
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto mb-8">
              {language === 'en' 
                ? "Join our exclusive Skool community where 20 minutes of daily learning transforms you into an AI researcher in just one year"
                : "加入我们专属的Skool社区，每日20分钟的学习让你在短短一年内转变为AI研究员"
              }
            </p>
            
            <div className="bg-gradient-to-r from-purple-600/20 to-blue-600/20 border border-purple-500/30 rounded-xl p-8 max-w-3xl mx-auto">
              <h3 className="text-2xl font-bold mb-4 text-purple-400">
                {language === 'en' ? 'Why 20 Minutes Works' : '为什么20分钟有效'}
              </h3>
              <p className="text-gray-300 mb-6">
                {language === 'en' 
                  ? "Research shows that consistent, focused learning in small increments is more effective than marathon study sessions. Our 20-minute daily approach ensures you stay engaged, retain information better, and build lasting habits that accelerate your research career."
                  : "研究表明，持续的小增量专注学习比马拉松式学习更有效。我们每日20分钟的方法确保你保持参与，更好地保留信息，并建立加速你研究生涯的持久习惯。"
                }
              </p>
              <div className="flex justify-center">
                <a 
                  href="https://skool.com/open-superintelligence-lab" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-medium rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all duration-200"
                >
                  {language === 'en' ? 'Join Skool Community' : '加入Skool社区'}
                </a>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
            {skoolFeatures.map((feature, index) => (
              <div key={index} className="p-6 border border-gray-800 rounded-xl bg-slate-800/30 hover:border-purple-500/50 transition-all duration-300">
                <div className="text-3xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold mb-3 text-purple-400">{feature.title}</h3>
                <p className="text-gray-300 text-sm leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Learning Path */}
        <section className="mb-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">
              {language === 'en' ? 'Your Learning Journey' : '你的学习之旅'}
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              {language === 'en' 
                ? "From complete beginner to contributing AI researcher in structured phases"
                : "从完全初学者到在结构化阶段贡献的AI研究员"
              }
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto">
            <div className="space-y-8">
              <div className="flex items-start gap-6 p-6 border border-gray-800 rounded-xl bg-slate-800/30">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold">
                  1
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2 text-cyan-400">
                    {language === 'en' ? 'Foundation (Months 1-3)' : '基础阶段（第1-3个月）'}
                  </h3>
                  <p className="text-gray-300">
                    {language === 'en' 
                      ? "Master the fundamentals: Python, mathematics, machine learning basics, and research methodology. Build your first AI models and understand the research landscape."
                      : "掌握基础：Python、数学、机器学习基础和研究方法论。构建你的第一个AI模型并了解研究领域。"
                    }
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-6 p-6 border border-gray-800 rounded-xl bg-slate-800/30">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold">
                  2
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2 text-purple-400">
                    {language === 'en' ? 'Specialization (Months 4-6)' : '专业化（第4-6个月）'}
                  </h3>
                  <p className="text-gray-300">
                    {language === 'en' 
                      ? "Choose your focus area: NLP, Computer Vision, or Reinforcement Learning. Deep dive into advanced techniques, read cutting-edge papers, and implement state-of-the-art models."
                      : "选择你的专业领域：NLP、计算机视觉或强化学习。深入高级技术，阅读前沿论文，并实现最先进的模型。"
                    }
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-6 p-6 border border-gray-800 rounded-xl bg-slate-800/30">
                <div className="w-12 h-12 bg-gradient-to-r from-pink-500 to-orange-500 rounded-full flex items-center justify-center text-white font-bold">
                  3
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2 text-pink-400">
                    {language === 'en' ? 'Research & Contribution (Months 7-9)' : '研究与贡献（第7-9个月）'}
                  </h3>
                  <p className="text-gray-300">
                    {language === 'en' 
                      ? "Start contributing to open-source projects, identify research problems, and begin your own experiments. Learn to write papers and present your findings."
                      : "开始为开源项目做出贡献，识别研究问题，并开始你自己的实验。学习撰写论文并展示你的发现。"
                    }
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-6 p-6 border border-gray-800 rounded-xl bg-slate-800/30">
                <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-yellow-500 rounded-full flex items-center justify-center text-white font-bold">
                  4
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2 text-orange-400">
                    {language === 'en' ? 'Mastery & Career (Months 10-12)' : '掌握与职业（第10-12个月）'}
                  </h3>
                  <p className="text-gray-300">
                    {language === 'en' 
                      ? "Lead research projects, mentor others, and establish yourself as a recognized AI researcher. Prepare for career opportunities at top AI labs and companies."
                      : "领导研究项目，指导他人，并确立自己作为公认的AI研究员的地位。为顶级AI实验室和公司的职业机会做准备。"
                    }
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="text-center">
          <div className="bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-cyan-600/20 border border-blue-500/30 rounded-xl p-8 max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold mb-4">
              {language === 'en' ? 'Ready to Start Your Journey?' : '准备好开始你的旅程了吗？'}
            </h2>
            <p className="text-gray-300 mb-8 text-lg">
              {language === 'en' 
                ? "Join thousands of aspiring AI researchers who are transforming their careers with just 20 minutes a day. Your future as an AI researcher starts today."
                : "加入数千名有抱负的AI研究员，他们每天仅用20分钟就在改变自己的职业生涯。你作为AI研究员的未来从今天开始。"
              }
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a 
                href="https://skool.com/open-superintelligence-lab" 
                target="_blank" 
                rel="noopener noreferrer"
                className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200"
              >
                {language === 'en' ? 'Join Skool Community' : '加入Skool社区'}
              </a>
              <Link 
                href="/research/deepseek"
                className="px-8 py-3 border border-gray-600/50 text-gray-300 font-medium rounded-lg hover:border-blue-500/50 hover:text-white transition-all duration-200"
              >
                {language === 'en' ? 'Explore Research Projects' : '探索研究项目'}
              </Link>
            </div>
          </div>
        </section>
      </main>
    </>
  );
}
