'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function Home() {
  const { language } = useLanguage();

  const getText = (en: string, zh: string) => {
    return language === 'zh' ? zh : en;
  };

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 min-h-screen flex flex-col">
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
        
        <div className="relative container mx-auto px-6 pt-32 pb-24 flex-grow flex items-center">
          <div className="text-center max-w-5xl mx-auto">
            {/* Enhanced title with more effects */}
            <div className="relative">
              <div className="flex flex-col items-center">
                <div className="relative text-center">
                  <h1 className="text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-medium mb-8 leading-tight">
                    <span className="bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent">
                      {getText('Open', '开放')}
                    </span>
                    <span className="ml-4 bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                      {getText('Superintelligence', '超级智能')}
                    </span>
                  </h1>
                  
                  {/* Glow effect for the entire title */}
                  <div className="absolute inset-0 text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-medium leading-tight blur-sm">
                    <span className="bg-gradient-to-r from-green-400/20 via-emerald-400/20 to-teal-400/20 bg-clip-text text-transparent">
                      {getText('Open', '开放')}
                    </span>
                    <span className="ml-4 bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-cyan-400/20 bg-clip-text text-transparent">
                      {getText('Superintelligence', '超级智能')}
                    </span>
                  </div>
                </div>
                
                {/* Subtitle */}
                <div className="relative mt-1 pb-6">
                  <h2 className="relative z-10 text-2xl md:text-3xl lg:text-4xl xl:text-5xl font-semibold bg-gradient-to-r from-amber-300 via-orange-400 to-rose-400 bg-clip-text text-transparent animate-pulse leading-relaxed">
                    {getText('The Most Difficult Project In Human History', '人类历史上最困难的项目')}
                  </h2>
                  {/* Glow effect for subtitle */}
                  <div className="absolute inset-0 text-2xl md:text-3xl lg:text-4xl xl:text-5xl font-semibold blur-lg opacity-50 pointer-events-none leading-relaxed">
                    <span className="bg-gradient-to-r from-amber-300 via-orange-400 to-rose-400 bg-clip-text text-transparent">
                      {getText('The Most Difficult Project In Human History', '人类历史上最困难的项目')}
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Enhanced decorative elements */}
              <div className="absolute -top-6 -left-6 w-10 h-10 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full opacity-70 animate-bounce shadow-lg shadow-green-500/30"></div>
              <div className="absolute top-1/2 -right-6 w-8 h-8 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full opacity-70 animate-bounce delay-500 shadow-lg shadow-purple-500/30"></div>
              <div className="absolute -bottom-6 left-1/2 w-6 h-6 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full opacity-70 animate-bounce delay-1000 shadow-lg shadow-blue-500/30"></div>
              
              {/* Additional floating elements */}
              <div className="absolute top-1/4 -left-8 w-3 h-3 bg-gradient-to-r from-pink-500 to-rose-500 rounded-full opacity-50 animate-pulse delay-300"></div>
              <div className="absolute top-3/4 -right-8 w-4 h-4 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full opacity-60 animate-pulse delay-700"></div>
              <div className="absolute -top-2 left-1/4 w-2 h-2 bg-gradient-to-r from-indigo-500 to-blue-500 rounded-full opacity-40 animate-pulse delay-900"></div>
              <div className="absolute -bottom-2 right-1/4 w-2.5 h-2.5 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-full opacity-55 animate-pulse delay-600"></div>
              
              {/* Rotating geometric shapes */}
              <div className="absolute top-1/6 -left-10 w-3 h-3 bg-gradient-to-r from-emerald-500/60 to-teal-500/60 rotate-45 animate-spin" style={{animationDuration: '8s'}}></div>
              <div className="absolute bottom-1/6 -right-10 w-2 h-4 bg-gradient-to-r from-purple-500/50 to-pink-500/50 rotate-12 animate-spin delay-1000" style={{animationDuration: '6s'}}></div>
              
              {/* Glowing rings */}
              <div className="absolute top-1/3 left-1/3 w-12 h-12 border-2 border-blue-400/30 rounded-full animate-pulse delay-400"></div>
              <div className="absolute bottom-1/3 right-1/3 w-8 h-8 border-2 border-purple-400/40 rounded-full animate-pulse delay-800"></div>
            </div>
            
            {/* Tags */}
            <div className="flex flex-wrap justify-center gap-4 text-sm text-slate-400 mt-4 mb-8">
              <span className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-full">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                {getText('Open Source', '开源')}
              </span>
              <span className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-full">
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse delay-300"></div>
                {getText('LLM Research', '大模型研究')}
              </span>
              <span className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-full">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse delay-700"></div>
                {getText('Innovation', '创新')}
              </span>
            </div>
            
            {/* Call to action buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <button 
                onClick={() => document.getElementById('research-projects')?.scrollIntoView({ behavior: 'smooth' })}
                className="group px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-blue-500/25"
              >
                <span className="flex items-center gap-2">
                  {getText('Explore & Participate', '探索并参与')}
                  <svg className="w-5 h-5 group-hover:translate-y-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                  </svg>
                </span>
              </button>
              <Link 
                href="/about" 
                className="group px-8 py-4 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-blue-500 hover:text-blue-400 transition-all duration-300 transform hover:scale-105"
              >
                <span className="flex items-center gap-2">
                  {getText('Learn More', '了解更多')}
                  <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </span>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Main Projects Section */}
      <main id="research-projects" className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {/* DeepSeek Sparse Attention Project */}
            <Link 
              href="/blog/deepseek-sparse-attention"
              className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-blue-500/50 hover:shadow-2xl hover:shadow-blue-500/10 transition-all duration-300"
            >
              <div className="absolute top-4 left-4">
                <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">{getText('Research', '研究')}</span>
              </div>
              <div className="absolute top-4 right-4">
                <span className="bg-blue-500/20 text-blue-400 text-xs px-2 py-1 rounded-md">{getText('Featured', '精选')}</span>
              </div>
              
              <div className="mt-8">
                <h4 className="text-xl font-bold mb-3 group-hover:text-blue-400 transition-colors">
                  DeepSeek Sparse Attention - DeepSeek-V3.2-Exp
                </h4>
                <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                  {getText(
                    'Advanced research on DeepSeek\'s innovative sparse attention mechanisms for efficient long-context processing and memory optimization',
                    'DeepSeek创新稀疏注意力机制的高级研究，用于高效长上下文处理和内存优化'
                  )}
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">DeepSeek Research</span>
                  <span className="text-blue-400 text-sm group-hover:text-blue-300 transition-colors">
                    {getText('Learn More', '了解更多')} →
                  </span>
                </div>
              </div>
            </Link>

            {/* Tiny Recursive Model Project */}
            <Link 
              href="/blog/tiny-recursive-model"
              className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-purple-500/50 hover:shadow-2xl hover:shadow-purple-500/10 transition-all duration-300"
            >
              <div className="absolute top-4 left-4">
                <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">{getText('Research', '研究')}</span>
              </div>
              <div className="absolute top-4 right-4">
                <span className="bg-purple-500/20 text-purple-400 text-xs px-2 py-1 rounded-md">{getText('Latest', '最新')}</span>
              </div>
              
              <div className="mt-8">
                <h4 className="text-xl font-bold mb-3 group-hover:text-purple-400 transition-colors">
                  {getText('Tiny Recursive Model', '微型递归模型')}
                </h4>
                <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                  {getText(
                    'How a 7M parameter model beats 100x bigger models at Sudoku, Mazes, and ARC-AGI using recursive reasoning with a 2-layer transformer',
                    '7M参数模型如何使用2层transformer的递归推理在数独、迷宫和ARC-AGI上击败100倍大的模型'
                  )}
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">{getText('AI Research', 'AI研究')}</span>
                  <span className="text-purple-400 text-sm group-hover:text-purple-300 transition-colors">
                    {getText('Learn More', '了解更多')} →
                  </span>
                </div>
              </div>
            </Link>

            {/* Pretrain LLM with NVFP4 Project */}
            <Link 
              href="/blog/pretrain-llm-with-nvfp4"
              className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-green-500/50 hover:shadow-2xl hover:shadow-green-500/10 transition-all duration-300"
            >
              <div className="absolute top-4 left-4">
                <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">{getText('Research', '研究')}</span>
              </div>
              <div className="absolute top-4 right-4">
                <span className="bg-green-500/20 text-green-400 text-xs px-2 py-1 rounded-md">{getText('Featured', '精选')}</span>
              </div>
              
              <div className="mt-8">
                <h4 className="text-xl font-bold mb-3 group-hover:text-green-400 transition-colors">
                  {getText('Pretrain LLM with NVFP4', '用NVFP4预训练LLM')}
                </h4>
                <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                  {getText(
                    'NVIDIA\'s breakthrough 4-bit training methodology achieving 2-3x speedup and 50% memory reduction without sacrificing model quality',
                    'NVIDIA突破性4位训练方法实现2-3倍加速和50%内存减少，同时不牺牲模型质量'
                  )}
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">NVIDIA Research</span>
                  <span className="text-green-400 text-sm group-hover:text-green-300 transition-colors">
                    {getText('Explore', '探索')} →
                  </span>
                </div>
              </div>
            </Link>
          </div>
        </div>
      </main>
    </>
  );
}
