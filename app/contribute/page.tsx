'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function Contribute() {
  const { language } = useLanguage();

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="relative container mx-auto px-6 pt-28 pb-8">
          <div className="max-w-6xl mx-auto">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex flex-wrap gap-3 justify-center">
                <a 
                  href="https://github.com/open-superintelligence-lab"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-600/50 hover:border-green-500/50 text-slate-300 hover:text-green-400 rounded-lg transition-all duration-200"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                  </svg>
                  GitHub
                </a>
                <a 
                  href="https://discord.com/invite/6AbXGpKTwN"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-600/50 hover:border-indigo-500/50 text-slate-300 hover:text-indigo-400 rounded-lg transition-all duration-200"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
                  </svg>
                  Discord
                </a>
                <a 
                  href="https://www.skool.com/become-ai-researcher-2669/about"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-600/50 hover:border-amber-500/50 text-slate-300 hover:text-amber-400 rounded-lg transition-all duration-200"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                  Skool
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
        <div className="container mx-auto px-6 max-w-6xl">
          
          {/* About Our Lab */}
          <div className="relative overflow-hidden bg-gradient-to-br from-slate-800/80 to-slate-700/80 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8 mb-8 shadow-xl">
            {/* Decorative elements */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-full blur-3xl"></div>
            <div className="absolute bottom-0 left-0 w-40 h-40 bg-gradient-to-tr from-emerald-500/10 to-cyan-500/10 rounded-full blur-3xl"></div>
            
            <div className="relative flex items-start gap-6">
              <div className="flex-shrink-0">
                <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center shadow-lg shadow-green-500/30">
                  <span className="text-3xl">🔬</span>
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-3xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent mb-4">
                  {language === 'en' ? 'About Open Superintelligence Lab' : '关于开放超级智能实验室'}
                </h2>
                <p className="text-slate-300 mb-4 leading-relaxed text-lg">
                  {language === 'en' 
                    ? 'We conduct open research on neural networks and LLMs. We bring top-tier company and university experience to everyone to contribute to AI research and accelerate the progress of humanity.'
                    : '我们对神经网络和大语言模型进行开放研究。我们将顶级公司和大学的经验带给每个人，让他们为AI研究做出贡献并加速人类进步。'
                  }
                </p>
              </div>
            </div>
          </div>

          {/* Featured Research Example */}
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
              <span className="text-3xl">✨</span>
              {language === 'en' ? 'Featured Research' : '精选研究'}
            </h2>
            <Link href="/blog/deepseek-sparse-attention">
              <div className="group relative overflow-hidden bg-gradient-to-br from-blue-900/40 to-purple-900/40 backdrop-blur-sm border border-blue-500/30 rounded-2xl p-8 hover:border-blue-400/60 transition-all duration-300 cursor-pointer shadow-lg hover:shadow-2xl hover:shadow-blue-500/20">
                {/* Animated gradient overlay */}
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-blue-500/10 to-blue-500/0 transform translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
                
                <div className="relative">
                  <div className="flex items-start gap-4 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
                      <span className="text-2xl">⚡</span>
                  </div>
                    <div className="flex-1">
                      <h3 className="text-2xl font-bold text-white mb-2 group-hover:text-blue-300 transition-colors">
                        DeepSeek Sparse Attention
                  </h3>
                      <p className="text-blue-300 text-sm mb-3">
                        From O(L²) to O(Lk) - The Lightning Indexer Breakthrough
                      </p>
              </div>
                  </div>
                  
                  <p className="text-slate-300 leading-relaxed mb-4">
                  {language === 'en' 
                      ? 'Our research on DeepSeek\'s sparse attention mechanism that reduces attention complexity from quadratic to near-linear scaling. We conducted experiments comparing standard attention vs sparse attention, showing up to 302% performance improvements on small LLMs.'
                      : '我们对DeepSeek稀疏注意力机制的研究，将注意力复杂度从二次缩放降低到近线性缩放。我们进行了标准注意力与稀疏注意力的对比实验，在小型LLM上显示出高达302%的性能改进。'
                  }
                </p>
                  
                  <div className="flex items-center gap-4 text-sm text-slate-400">
                    <span className="flex items-center gap-1">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      {language === 'en' ? 'Technical Deep Dive' : '技术深入分析'}
                    </span>
                    <span className="text-slate-600">•</span>
                    <span className="flex items-center gap-1">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                      {language === 'en' ? 'Research Article' : '研究文章'}
                    </span>
              </div>

                  <div className="mt-6 flex items-center gap-2 text-blue-400 group-hover:text-blue-300 transition-colors">
                    <span className="font-semibold">{language === 'en' ? 'Read Full Research' : '阅读完整研究'}</span>
                    <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                    </svg>
                  </div>
                </div>
              </div>
            </Link>
          </div>

          {/* How to Contribute */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6 text-center">
              {language === 'en' ? 'How to Contribute' : '如何贡献'}
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Step 1: Choose Project */}
              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-white">1</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-3">
                  {language === 'en' ? 'Choose a Project' : '选择项目'}
                </h3>
                <p className="text-slate-300 text-sm leading-relaxed">
                  {language === 'en' 
                    ? 'Browse our research projects above and select one that aligns with your interests and expertise.'
                    : '浏览我们上面的研究项目，选择与您的兴趣和专业知识相符的项目。'
                  }
                </p>
              </div>

              {/* Step 2: Create Issues */}
              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-white">2</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-3">
                  {language === 'en' ? 'Create GitHub Issues' : '创建 GitHub 问题'}
                </h3>
                  <p className="text-slate-300 text-sm leading-relaxed">
                    {language === 'en' 
                      ? 'Create issues to discuss ideas, bugs, improvements, or new research directions. We value meaningful contributions that advance our research goals, not trivial ones like "Fixed typo".'
                      : '创建问题来讨论想法、报告错误、建议改进或提出新的研究方向。我们重视推进我们研究目标的有意义的贡献，而不是像"修复拼写错误"这样的琐碎修复。'
                    }
                </p>
              </div>

              {/* Step 3: Collaborate */}
              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-white">3</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-3">
                  {language === 'en' ? 'Discuss & Collaborate' : '讨论与协作'}
                </h3>
                <p className="text-slate-300 text-sm leading-relaxed">
                  {language === 'en' 
                    ? 'Engage in discussions, share research findings, and collaborate with the community to advance the projects.'
                    : '参与讨论，分享研究发现，与社区协作推进项目发展。'
                  }
                </p>
              </div>
            </div>
          </div>

          {/* Contribution Guidelines */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">
              {language === 'en' ? 'Contribution Guidelines' : '贡献指南'}
            </h2>
            
            <div className="space-y-6">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                  <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {language === 'en' ? 'Research & Ideation' : '研究与构思'}
                  </h3>
                  <p className="text-slate-300 leading-relaxed">
                    {language === 'en' 
                      ? 'Share your research ideas, experimental results, and theoretical insights. We welcome contributions from all levels of expertise.'
                      : '分享您的研究想法、实验结果和理论见解。我们欢迎所有专业水平的贡献。'
                    }
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {language === 'en' ? 'Open Discussion' : '开放讨论'}
                  </h3>
                  <p className="text-slate-300 leading-relaxed">
                    {language === 'en' 
                      ? 'Use GitHub issues and discussions to engage with the community. Join our Discord server for real-time conversations, ask questions, share knowledge, and help others learn.'
                      : '使用 GitHub 问题和讨论与社区互动。加入我们的 Discord 服务器进行实时对话，提出问题，分享知识，帮助他人学习。'
                    }
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                  <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {language === 'en' ? 'Code & Implementation' : '代码与实现'}
                  </h3>
                  <p className="text-slate-300 leading-relaxed">
                    {language === 'en' 
                      ? 'Contribute code, documentation, and implementations. Follow our coding standards and submit pull requests for review.'
                      : '贡献代码、文档和实现。遵循我们的编码标准并提交拉取请求以供审查。'
                    }
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-8 h-8 bg-cyan-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                  <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {language === 'en' ? 'Documentation & Learning' : '文档与学习'}
                  </h3>
                  <p className="text-slate-300 leading-relaxed">
                    {language === 'en' 
                      ? 'Help improve documentation, create tutorials, and develop learning materials to make our research more accessible.'
                      : '帮助改进文档，创建教程，开发学习材料，使我们的研究更容易获得。'
                    }
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Call to Action */}
          <div className="text-center">
            <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 rounded-xl p-8">
              <h2 className="text-2xl font-bold text-white mb-4">
                {language === 'en' ? 'Ready to Contribute?' : '准备好贡献了吗？'}
              </h2>
              <p className="text-slate-300 mb-6 leading-relaxed">
                {language === 'en' 
                  ? 'Join our community and help shape the future of AI research. Every contribution, no matter how small, makes a difference.'
                  : '加入我们的社区，帮助塑造AI研究的未来。每一个贡献，无论多么微小，都会产生影响。'
                }
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <a 
                  href="https://github.com/open-superintelligence-lab"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white font-semibold rounded-xl hover:from-green-700 hover:to-emerald-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-green-500/25"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                  </svg>
                  {language === 'en' ? 'GitHub' : 'GitHub'}
                </a>
                <a 
                  href="https://discord.com/invite/6AbXGpKTwN"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold rounded-xl hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-indigo-500/25"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
                  </svg>
                  {language === 'en' ? 'Discord' : 'Discord'}
                </a>
                <a 
                  href="https://www.skool.com/become-ai-researcher-2669/about"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-gradient-to-r from-amber-600 to-orange-600 text-white font-semibold rounded-xl hover:from-amber-700 hover:to-orange-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-amber-500/25"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                  {language === 'en' ? 'Skool' : 'Skool'}
                </a>
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
