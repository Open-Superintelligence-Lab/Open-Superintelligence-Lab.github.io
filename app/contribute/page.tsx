'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function Contribute() {
  const { language } = useLanguage();

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Background effects */}
        <div className="absolute inset-0 bg-gradient-to-r from-green-600/20 via-emerald-600/20 to-teal-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-green-500/5 to-transparent"></div>
        </div>
        
        {/* Animated background particles */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/6 left-1/6 w-3 h-3 bg-gradient-to-r from-green-400 to-emerald-400 rounded-full opacity-60 animate-pulse"></div>
          <div className="absolute top-1/4 right-1/5 w-2 h-2 bg-gradient-to-r from-teal-400 to-cyan-400 rounded-full opacity-50 animate-pulse delay-300"></div>
          <div className="absolute top-1/3 left-1/8 w-4 h-4 bg-gradient-to-r from-emerald-400 to-green-400 rounded-full opacity-40 animate-pulse delay-700"></div>
          <div className="absolute bottom-1/4 right-1/6 w-2.5 h-2.5 bg-gradient-to-r from-cyan-400 to-teal-400 rounded-full opacity-55 animate-pulse delay-1000"></div>
        </div>
        
        <div className="relative container mx-auto px-6 py-24">
          <div className="text-center max-w-4xl mx-auto">
            <div className="relative">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium mb-8 leading-tight">
                <span className="bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent">
                  {language === 'en' ? 'Contribute to Our Lab' : '为我们的实验室贡献'}
                </span>
              </h1>
              
              {/* Glow effect for the title */}
              <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-medium leading-tight blur-sm">
                <span className="bg-gradient-to-r from-green-400/20 via-emerald-400/20 to-teal-400/20 bg-clip-text text-transparent">
                  {language === 'en' ? 'Contribute to Our Lab' : '为我们的实验室贡献'}
                </span>
              </div>
            </div>
            
            <p className="text-xl text-slate-300 mb-12 leading-relaxed">
              {language === 'en' 
                ? 'Join our open-source research community and help advance the field of artificial intelligence and superintelligence'
                : '加入我们的开源研究社区，帮助推进人工智能和超级智能领域的发展'
              }
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
        <div className="container mx-auto px-6 max-w-6xl">
          
          {/* About Our Lab */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0">
                <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                  </svg>
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-bold text-white mb-4">
                  {language === 'en' ? 'About Open Superintelligence Lab' : '关于开放超级智能实验室'}
                </h2>
                <p className="text-slate-300 mb-6 leading-relaxed">
                  {language === 'en' 
                    ? 'We are a collaborative research community focused on advancing artificial intelligence and superintelligence through open-source projects, cutting-edge research, and knowledge sharing. Our lab brings together researchers, developers, and enthusiasts to work on innovative AI technologies and methodologies.'
                    : '我们是一个协作研究社区，专注于通过开源项目、前沿研究和知识共享来推进人工智能和超级智能。我们的实验室汇集了研究人员、开发人员和爱好者，致力于创新的AI技术和方法。'
                  }
                </p>
                <p className="text-slate-300 leading-relaxed">
                  {language === 'en' 
                    ? 'We believe in the power of open collaboration and transparent research to accelerate progress in AI. Our projects span from foundational learning resources to advanced research in sparse attention mechanisms, mixture of experts, and other cutting-edge AI architectures.'
                    : '我们相信开放协作和透明研究的力量，能够加速AI的进步。我们的项目涵盖从基础学习资源到稀疏注意力机制、专家混合和其他前沿AI架构的高级研究。'
                  }
                </p>
              </div>
            </div>
          </div>

          {/* Our Projects */}
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-white mb-6 text-center">
              {language === 'en' ? 'Our Research Projects' : '我们的研究项目'}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Zero to AI Researcher */}
              <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white">
                    {language === 'en' ? 'Zero to AI Researcher' : '从零到AI研究员'}
                  </h3>
                </div>
                <p className="text-slate-300 text-sm leading-relaxed mb-4">
                  {language === 'en' 
                    ? 'Comprehensive learning path for becoming an AI researcher, covering foundational concepts to cutting-edge methodologies.'
                    : '成为AI研究员的综合学习路径，涵盖从基础概念到前沿方法。'
                  }
                </p>
                <Link 
                  href="/zero-to-ai-researcher"
                  className="text-green-400 text-sm hover:text-green-300 transition-colors"
                >
                  {language === 'en' ? 'Learn More →' : '了解更多 →'}
                </Link>
              </div>

              {/* DeepSeek Sparse Attention */}
              <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white">
                    {language === 'en' ? 'DeepSeek Sparse Attention' : 'DeepSeek 稀疏注意力'}
                  </h3>
                </div>
                <p className="text-slate-300 text-sm leading-relaxed mb-4">
                  {language === 'en' 
                    ? 'Advanced research on sparse attention mechanisms for efficient long-context processing and memory optimization.'
                    : '稀疏注意力机制的高级研究，用于高效的长上下文处理和内存优化。'
                  }
                </p>
                <Link 
                  href="/deepseek-sparse-attention"
                  className="text-blue-400 text-sm hover:text-blue-300 transition-colors"
                >
                  {language === 'en' ? 'Learn More →' : '了解更多 →'}
                </Link>
              </div>

              {/* GLM4-MoE */}
              <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white">
                    {language === 'en' ? 'GLM4-MoE' : 'GLM4-MoE'}
                  </h3>
                </div>
                <p className="text-slate-300 text-sm leading-relaxed mb-4">
                  {language === 'en' 
                    ? 'Mixture of Experts implementation with GLM4 architecture for efficient scaling and improved performance.'
                    : 'GLM4架构的专家混合实现，用于高效扩展和性能提升。'
                  }
                </p>
                <a 
                  href="https://github.com/THUDM/GLM-4"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 text-sm hover:text-purple-300 transition-colors"
                >
                  {language === 'en' ? 'Explore →' : '探索 →'}
                </a>
              </div>

              {/* DeepSeek + GLM4 Hybrid */}
              <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-8 h-8 bg-cyan-500/20 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white">
                    {language === 'en' ? 'DeepSeek + GLM4 Hybrid' : 'DeepSeek + GLM4 混合'}
                  </h3>
                </div>
                <p className="text-slate-300 text-sm leading-relaxed mb-4">
                  {language === 'en' 
                    ? 'Innovative combination of DeepSeek\'s sparse attention with GLM4\'s Mixture of Experts architecture.'
                    : 'DeepSeek稀疏注意力与GLM4专家混合架构的创新结合。'
                  }
                </p>
                <a 
                  href="https://github.com/Open-Superintelligence-Lab/deepseek-attention-glm4-moe"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-cyan-400 text-sm hover:text-cyan-300 transition-colors"
                >
                  {language === 'en' ? 'Learn More →' : '了解更多 →'}
                </a>
              </div>
            </div>
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
                  className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white font-semibold rounded-xl hover:from-green-700 hover:to-emerald-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-green-500/25"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                  </svg>
                  {language === 'en' ? 'Visit Our GitHub' : '访问我们的 GitHub'}
                </a>
                <a 
                  href="https://discord.com/invite/6AbXGpKTwN"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold rounded-xl hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-indigo-500/25"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
                  </svg>
                  {language === 'en' ? 'Join Our Discord' : '加入我们的 Discord'}
                </a>
                <Link 
                  href="/"
                  className="inline-flex items-center gap-2 px-8 py-4 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-green-500 hover:text-green-400 transition-all duration-300 transform hover:scale-105"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                  </svg>
                  {language === 'en' ? 'Back to Home' : '返回首页'}
                </Link>
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
