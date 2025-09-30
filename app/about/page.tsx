'use client';

import { useLanguage } from "@/components/providers/language-provider";

export default function AboutPage() {
  const { language } = useLanguage();

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Enhanced background effects */}
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
        </div>
        
        {/* Enhanced animated background particles */}
        <div className="absolute inset-0 overflow-hidden">
          {/* Large floating particles - some made more glowy and dreamy */}
          <div className="absolute top-1/6 left-1/6 w-3 h-3 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full opacity-60 animate-pulse"></div>
          <div className="absolute top-1/4 right-1/5 w-2 h-2 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full opacity-50 animate-pulse delay-300 shadow-lg shadow-purple-400/40"></div>
          <div className="absolute top-1/3 left-1/8 w-4 h-4 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full opacity-40 animate-pulse delay-700 blur-sm shadow-lg shadow-emerald-400/30"></div>
          <div className="absolute bottom-1/4 right-1/6 w-2.5 h-2.5 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full opacity-55 animate-pulse delay-1000"></div>
          <div className="absolute bottom-1/3 left-1/4 w-1.5 h-1.5 bg-gradient-to-r from-pink-400 to-purple-400 rounded-full opacity-60 animate-pulse delay-500 blur-sm shadow-lg shadow-pink-400/35"></div>
          <div className="absolute top-2/3 right-1/3 w-3.5 h-3.5 bg-gradient-to-r from-teal-400 to-emerald-400 rounded-full opacity-35 animate-pulse delay-1200 blur-sm shadow-lg shadow-teal-400/25"></div>
          
          {/* Medium particles - enhanced with glow */}
          <div className="absolute top-1/5 left-2/5 w-1 h-1 bg-blue-400/60 rounded-full animate-pulse delay-200 shadow-lg shadow-blue-400/50"></div>
          <div className="absolute top-2/5 right-2/5 w-1.5 h-1.5 bg-purple-400/50 rounded-full animate-pulse delay-800 blur-sm shadow-lg shadow-purple-400/45"></div>
          <div className="absolute bottom-2/5 left-3/5 w-1 h-1 bg-cyan-400/70 rounded-full animate-pulse delay-400"></div>
          <div className="absolute bottom-1/5 right-1/5 w-1.5 h-1.5 bg-pink-400/45 rounded-full animate-pulse delay-900 blur-sm shadow-lg shadow-pink-400/40"></div>
          <div className="absolute top-3/5 left-1/5 w-1 h-1 bg-emerald-400/65 rounded-full animate-pulse delay-600 shadow-lg shadow-emerald-400/55"></div>
          
          {/* Small twinkling particles - some made more dreamy */}
          <div className="absolute top-1/8 left-3/8 w-0.5 h-0.5 bg-white/80 rounded-full animate-pulse delay-150 blur-sm shadow-lg shadow-white/60"></div>
          <div className="absolute top-1/7 right-3/8 w-0.5 h-0.5 bg-blue-300/90 rounded-full animate-pulse delay-750"></div>
          <div className="absolute bottom-1/8 left-2/8 w-0.5 h-0.5 bg-purple-300/85 rounded-full animate-pulse delay-350 blur-sm shadow-lg shadow-purple-300/70"></div>
          <div className="absolute bottom-1/7 right-2/8 w-0.5 h-0.5 bg-cyan-300/80 rounded-full animate-pulse delay-950"></div>
          <div className="absolute top-4/5 left-4/8 w-0.5 h-0.5 bg-pink-300/75 rounded-full animate-pulse delay-550 blur-sm shadow-lg shadow-pink-300/65"></div>
          <div className="absolute top-3/8 right-1/8 w-0.5 h-0.5 bg-emerald-300/85 rounded-full animate-pulse delay-1150"></div>
          
          {/* Floating geometric shapes - some made more ethereal */}
          <div className="absolute top-1/6 right-1/8 w-2 h-2 bg-gradient-to-r from-blue-500/30 to-purple-500/30 rotate-45 animate-pulse delay-250 blur-sm shadow-lg shadow-blue-500/25"></div>
          <div className="absolute bottom-1/6 left-1/8 w-1.5 h-1.5 bg-gradient-to-r from-cyan-500/40 to-pink-500/40 rotate-12 animate-pulse delay-650"></div>
          <div className="absolute top-1/2 right-1/6 w-1 h-3 bg-gradient-to-r from-emerald-500/35 to-teal-500/35 rotate-75 animate-pulse delay-850 blur-sm shadow-lg shadow-emerald-500/20"></div>
          
          {/* Enhanced glowing orbs - made more dreamy */}
          <div className="absolute top-1/4 left-1/2 w-6 h-6 bg-gradient-to-r from-blue-400/20 to-purple-400/20 rounded-full blur-sm animate-pulse delay-450 shadow-lg shadow-blue-400/15"></div>
          <div className="absolute bottom-1/4 right-1/2 w-4 h-4 bg-gradient-to-r from-cyan-400/25 to-pink-400/25 rounded-full blur-sm animate-pulse delay-1050 shadow-lg shadow-cyan-400/20"></div>
          <div className="absolute top-1/2 left-1/3 w-5 h-5 bg-gradient-to-r from-emerald-400/15 to-teal-400/15 rounded-full blur-sm animate-pulse delay-750 shadow-lg shadow-emerald-400/12"></div>
          
          {/* Additional dreamy particles */}
          <div className="absolute top-1/5 right-1/4 w-2 h-2 bg-gradient-to-r from-violet-400/30 to-fuchsia-400/30 rounded-full blur-sm animate-pulse delay-1100 shadow-lg shadow-violet-400/25"></div>
          <div className="absolute bottom-1/5 left-2/5 w-1.5 h-1.5 bg-gradient-to-r from-amber-400/35 to-orange-400/35 rounded-full blur-sm animate-pulse delay-550 shadow-lg shadow-amber-400/30"></div>
          <div className="absolute top-2/5 right-1/5 w-1 h-1 bg-gradient-to-r from-rose-400/40 to-pink-400/40 rounded-full blur-sm animate-pulse delay-850 shadow-lg shadow-rose-400/35"></div>
        </div>

        <div className="relative container mx-auto px-6 pt-32 pb-12">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-6">
              {language === 'en' ? 'About Our Lab' : '关于我们实验室'}
            </h1>
            <p className="text-xl text-slate-300 leading-relaxed">
              {language === 'en' 
                ? 'Advancing AI research through open collaboration and innovation'
                : '通过开放协作和创新推进AI研究'
              }
            </p>
          </div>
        </div>
        </div>
      </section>

      <main className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
        <div className="container mx-auto px-6">
          <div className="max-w-4xl mx-auto">
          <div className="grid gap-8 md:gap-12">
            {/* Mission Section */}
            <section className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-8 border border-slate-700/50">
              <h2 className="text-2xl font-bold text-blue-400 mb-6">
                {language === 'en' ? 'Our Mission' : '我们的使命'}
              </h2>
              <div className="space-y-4 text-slate-300 leading-relaxed">
                <p>
                  {language === 'en' 
                    ? 'At the Open Superintelligence Lab, we conduct open research on the best open source projects and Large Language Models (LLMs). Our mission is to advance the field of artificial intelligence through transparent, collaborative research that benefits the entire AI community.'
                    : '在开放超级智能实验室，我们对最好的开源项目和大语言模型(LLMs)进行开放研究。我们的使命是通过透明、协作的研究推进人工智能领域，造福整个AI社区。'
                  }
                </p>
                <p>
                  {language === 'en' 
                    ? 'We believe that the future of AI should be built on open principles, where knowledge is shared freely and innovations are accessible to everyone. Our research focuses on understanding, improving, and advancing the state-of-the-art in open source AI technologies.'
                    : '我们相信AI的未来应该建立在开放原则之上，知识自由分享，创新对所有人开放。我们的研究专注于理解、改进和推进开源AI技术的最先进水平。'
                  }
                </p>
              </div>
            </section>

            {/* Research Focus */}
            <section className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-8 border border-slate-700/50">
              <h2 className="text-2xl font-bold text-purple-400 mb-6">
                {language === 'en' ? 'Research Focus' : '研究重点'}
              </h2>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-blue-300">
                    {language === 'en' ? 'Open Source Projects' : '开源项目'}
                  </h3>
                  <p className="text-slate-300 leading-relaxed">
                    {language === 'en' 
                      ? 'We analyze and contribute to the most promising open source AI projects, identifying best practices and areas for improvement.'
                      : '我们分析并贡献最有前景的开源AI项目，识别最佳实践和改进领域。'
                    }
                  </p>
                </div>
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-purple-300">
                    {language === 'en' ? 'Large Language Models' : '大语言模型'}
                  </h3>
                  <p className="text-slate-300 leading-relaxed">
                    {language === 'en' 
                      ? 'We conduct research on state-of-the-art LLMs, exploring their capabilities, limitations, and potential for advancement.'
                      : '我们对最先进的大语言模型进行研究，探索它们的能力、局限性和改进潜力。'
                    }
                  </p>
                </div>
              </div>
            </section>

            {/* Current Projects */}
            <section className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-8 border border-slate-700/50">
              <h2 className="text-2xl font-bold text-green-400 mb-6">
                {language === 'en' ? 'Current Research Areas' : '当前研究领域'}
              </h2>
              <div className="space-y-6">
                <div className="border-l-4 border-blue-500 pl-6">
                  <h3 className="text-lg font-semibold text-blue-300 mb-2">
                    DeepSeek-V3.2-Exp Research
                  </h3>
                  <p className="text-slate-300 leading-relaxed">
                    {language === 'en' 
                      ? 'Investigating DeepSeek\'s Sparse Attention (DSA) mechanisms and long-context efficiency improvements in open source language models.'
                      : '研究DeepSeek的稀疏注意力机制(DSA)和开源语言模型中的长上下文效率改进。'
                    }
                  </p>
                </div>
                <div className="border-l-4 border-purple-500 pl-6">
                  <h3 className="text-lg font-semibold text-purple-300 mb-2">
                    GPT-OSS Research
                  </h3>
                  <p className="text-slate-300 leading-relaxed">
                    {language === 'en' 
                      ? 'Exploring OpenAI\'s open-source Mixture of Experts (MoE) language models with advanced reasoning capabilities and safety features.'
                      : '探索OpenAI的开源专家混合(MoE)语言模型，具有先进的推理能力和安全特性。'
                    }
                  </p>
                </div>
              </div>
            </section>

            {/* Values */}
            <section className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-8 border border-slate-700/50">
              <h2 className="text-2xl font-bold text-yellow-400 mb-6">
                {language === 'en' ? 'Our Values' : '我们的价值观'}
              </h2>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center space-y-3">
                  <div className="w-12 h-12 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto">
                    <span className="text-blue-400 text-xl">🔓</span>
                  </div>
                  <h3 className="font-semibold text-blue-300">
                    {language === 'en' ? 'Openness' : '开放性'}
                  </h3>
                  <p className="text-sm text-slate-400">
                    {language === 'en' 
                      ? 'Transparent research and open collaboration'
                      : '透明研究和开放协作'
                    }
                  </p>
                </div>
                <div className="text-center space-y-3">
                  <div className="w-12 h-12 bg-purple-500/20 rounded-full flex items-center justify-center mx-auto">
                    <span className="text-purple-400 text-xl">🚀</span>
                  </div>
                  <h3 className="font-semibold text-purple-300">
                    {language === 'en' ? 'Innovation' : '创新'}
                  </h3>
                  <p className="text-sm text-slate-400">
                    {language === 'en' 
                      ? 'Pushing the boundaries of AI research'
                      : '推动AI研究的边界'
                    }
                  </p>
                </div>
                <div className="text-center space-y-3">
                  <div className="w-12 h-12 bg-green-500/20 rounded-full flex items-center justify-center mx-auto">
                    <span className="text-green-400 text-xl">🤝</span>
                  </div>
                  <h3 className="font-semibold text-green-300">
                    {language === 'en' ? 'Collaboration' : '协作'}
                  </h3>
                  <p className="text-sm text-slate-400">
                    {language === 'en' 
                      ? 'Building a stronger AI community together'
                      : '共同建设更强大的AI社区'
                    }
                  </p>
                </div>
              </div>
            </section>

            {/* Call to Action */}
            <section className="text-center bg-gradient-to-r from-blue-600/20 to-purple-600/20 rounded-2xl p-8 border border-blue-500/30">
              <h2 className="text-2xl font-bold text-white mb-4">
                {language === 'en' ? 'Join Our Research' : '加入我们的研究'}
              </h2>
              <p className="text-slate-300 mb-6 max-w-2xl mx-auto">
                {language === 'en' 
                  ? 'Interested in contributing to open AI research? Explore our projects and learn how you can get involved in advancing the field of artificial intelligence.'
                  : '有兴趣为开放AI研究做贡献吗？探索我们的项目，了解如何参与推进人工智能领域的发展。'
                }
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <a 
                  href="/research" 
                  className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200"
                >
                  {language === 'en' ? 'View Research' : '查看研究'}
                </a>
                <a 
                  href="https://github.com/open-superintelligence-lab" 
                  className="px-6 py-3 border border-slate-600 text-slate-300 font-medium rounded-lg hover:border-blue-500 hover:text-blue-400 transition-all duration-200"
                  target="_blank" 
                  rel="noopener noreferrer"
                >
                  {language === 'en' ? 'GitHub' : 'GitHub'}
                </a>
              </div>
            </section>
          </div>
          </div>
        </div>
      </main>
    </>
  );
}
