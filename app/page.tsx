'use client';

import Link from "next/link";
import { Navigation } from "@/components/navigation";
import { useLanguage } from "@/components/providers/language-provider";
import { translations } from "@/lib/language-detection";

export default function Home() {
  const { language } = useLanguage();
  const t = translations[language];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <Navigation />
      
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        {/* Enhanced background effects */}
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
        </div>
        
        {/* Animated background particles */}
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-blue-400/30 rounded-full animate-pulse"></div>
          <div className="absolute top-1/3 right-1/3 w-1 h-1 bg-purple-400/40 rounded-full animate-pulse delay-300"></div>
          <div className="absolute bottom-1/3 left-1/3 w-1.5 h-1.5 bg-cyan-400/30 rounded-full animate-pulse delay-700"></div>
          <div className="absolute top-2/3 right-1/4 w-1 h-1 bg-pink-400/40 rounded-full animate-pulse delay-1000"></div>
        </div>
        
        <div className="relative container mx-auto px-6 py-24">
          <div className="text-center max-w-5xl mx-auto">
            {/* Enhanced title with more effects */}
            <div className="relative">
              <div className="flex flex-col items-center">
                <div className="relative">
                  <h1 className="text-5xl md:text-6xl lg:text-7xl font-black mb-2 bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent leading-tight">
                    {language === 'en' ? 'Open' : '开放'}
                  </h1>
                  {/* Glow effect for Open */}
                  <div className="absolute inset-0 text-5xl md:text-6xl lg:text-7xl font-black bg-gradient-to-r from-green-400/20 via-emerald-400/20 to-teal-400/20 bg-clip-text text-transparent blur-sm leading-tight">
                    {language === 'en' ? 'Open' : '开放'}
                  </div>
                </div>
                
                <div className="relative">
                  <h2 className="text-4xl md:text-5xl lg:text-6xl font-black mb-8 bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent leading-tight">
                    {language === 'en' ? 'Superintelligence Lab' : '超级智能实验室'}
                  </h2>
                  {/* Glow effect for main title */}
                  <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-black bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-cyan-400/20 bg-clip-text text-transparent blur-sm leading-tight">
                    {language === 'en' ? 'Superintelligence Lab' : '超级智能实验室'}
                  </div>
                </div>
              </div>
              
              {/* Decorative elements */}
              <div className="absolute -top-4 -left-4 w-8 h-8 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full opacity-60 animate-bounce"></div>
              <div className="absolute top-1/2 -right-4 w-6 h-6 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full opacity-60 animate-bounce delay-500"></div>
              <div className="absolute -bottom-4 left-1/2 w-4 h-4 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full opacity-60 animate-bounce delay-1000"></div>
            </div>
            
            {/* Enhanced tagline */}
            <div className="mt-8 mb-12">
              <div className="space-y-4">
                <p className="text-xl md:text-2xl text-slate-300 font-medium">
                  {language === 'en' ? (
                    <>
                      <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent font-semibold">
                        Pushing the boundaries
                      </span>{" "}
                      of artificial intelligence through{" "}
                      <span className="bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent font-semibold">
                        open research
                      </span>
                    </>
                  ) : (
                    <>
                      <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent font-semibold">
                        推动边界
                      </span>{" "}
                      通过{" "}
                      <span className="bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent font-semibold">
                        开放研究
                      </span>{" "}
                      推进人工智能
                    </>
                  )}
                </p>
                <div className="flex flex-wrap justify-center gap-4 text-sm text-slate-400">
                  <span className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-full">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                    {language === 'en' ? 'Open Source' : '开源'}
                  </span>
                  <span className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-full">
                    <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse delay-300"></div>
                    {language === 'en' ? 'LLM Research' : '大模型研究'}
                  </span>
                  <span className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-full">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse delay-700"></div>
                    {language === 'en' ? 'Innovation' : '创新'}
                  </span>
                </div>
              </div>
            </div>
            
            {/* Call to action buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <button 
                onClick={() => document.getElementById('research-projects')?.scrollIntoView({ behavior: 'smooth' })}
                className="group px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-blue-500/25"
              >
                <span className="flex items-center gap-2">
                  Explore Research
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
                  Learn More
                  <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </span>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Projects Section */}
      <main id="research-projects" className="container mx-auto px-6 py-12">
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
          {/* DeepSeek Project */}
          <Link 
            href="/research/deepseek"
            className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-blue-500/50 hover:shadow-2xl hover:shadow-blue-500/10 transition-all duration-300"
          >
            <div className="absolute top-4 left-4">
              <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">Research Hub</span>
            </div>
            <div className="absolute top-4 right-4">
              <span className="bg-green-500/20 text-green-400 text-xs px-2 py-1 rounded-md">Active</span>
            </div>
            
            <div className="mt-8">
              <h4 className="text-xl font-bold mb-3 group-hover:text-blue-400 transition-colors">
                DeepSeek Research
              </h4>
              <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                Comprehensive research hub featuring multiple DeepSeek projects including V3.2-Exp, healthcare applications, and security research
              </p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">Multiple Research Projects</span>
                <span className="text-blue-400 text-sm group-hover:text-blue-300 transition-colors">
                  Explore →
                </span>
              </div>
            </div>
          </Link>

          {/* GPT-OSS Project */}
          <Link 
            href="/research/gpt-oss"
            className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-purple-500/50 hover:shadow-2xl hover:shadow-purple-500/10 transition-all duration-300"
          >
            <div className="absolute top-4 left-4">
              <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">Model</span>
            </div>
            <div className="absolute top-4 right-4">
              <span className="bg-purple-500/20 text-purple-400 text-xs px-2 py-1 rounded-md">Open Source</span>
            </div>
            
            <div className="mt-8">
              <h4 className="text-xl font-bold mb-3 group-hover:text-purple-400 transition-colors">
                GPT-OSS Research
              </h4>
              <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                OpenAI's open-source MoE language models with advanced reasoning capabilities and safety features
              </p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">OpenAI Research</span>
                <span className="text-purple-400 text-sm group-hover:text-purple-300 transition-colors">
                  Learn More →
                </span>
              </div>
            </div>
          </Link>

          {/* Kimi K2 Project */}
          <Link 
            href="#"
            className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-orange-500/50 hover:shadow-2xl hover:shadow-orange-500/10 transition-all duration-300"
          >
            <div className="absolute top-4 left-4">
              <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">Model</span>
            </div>
            <div className="absolute top-4 right-4">
              <span className="bg-orange-500/20 text-orange-400 text-xs px-2 py-1 rounded-md">New</span>
            </div>
            
            <div className="mt-8">
              <h4 className="text-xl font-bold mb-3 group-hover:text-orange-400 transition-colors">
                Kimi K2
              </h4>
              <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                Moonshot AI's latest multimodal model with enhanced reasoning and code generation capabilities
              </p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">Moonshot AI</span>
                <span className="text-orange-400 text-sm group-hover:text-orange-300 transition-colors">
                  Coming Soon →
                </span>
              </div>
            </div>
          </Link>

          {/* NVIDIA LLM Project */}
          <Link 
            href="#"
            className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-green-500/50 hover:shadow-2xl hover:shadow-green-500/10 transition-all duration-300"
          >
            <div className="absolute top-4 left-4">
              <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">Model</span>
            </div>
            <div className="absolute top-4 right-4">
              <span className="bg-green-500/20 text-green-400 text-xs px-2 py-1 rounded-md">Hot</span>
            </div>
            
            <div className="mt-8">
              <h4 className="text-xl font-bold mb-3 group-hover:text-green-400 transition-colors">
                NVIDIA LLM
              </h4>
              <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                NVIDIA's enterprise-grade large language models optimized for GPU acceleration and deployment
              </p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">NVIDIA Research</span>
                <span className="text-green-400 text-sm group-hover:text-green-300 transition-colors">
                  Coming Soon →
                </span>
              </div>
            </div>
          </Link>

          {/* Meta Llama Project */}
          <Link 
            href="#"
            className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-cyan-500/50 hover:shadow-2xl hover:shadow-cyan-500/10 transition-all duration-300"
          >
            <div className="absolute top-4 left-4">
              <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">Model</span>
            </div>
            <div className="absolute top-4 right-4">
              <span className="bg-cyan-500/20 text-cyan-400 text-xs px-2 py-1 rounded-md">Popular</span>
            </div>
            
            <div className="mt-8">
              <h4 className="text-xl font-bold mb-3 group-hover:text-cyan-400 transition-colors">
                Meta Llama 3.1
              </h4>
              <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                Meta's latest open-source language model with improved instruction following and safety features
              </p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">Meta Research</span>
                <span className="text-cyan-400 text-sm group-hover:text-cyan-300 transition-colors">
                  Coming Soon →
                </span>
              </div>
            </div>
          </Link>

          {/* Anthropic Claude Project */}
          <Link 
            href="#"
            className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-pink-500/50 hover:shadow-2xl hover:shadow-pink-500/10 transition-all duration-300"
          >
            <div className="absolute top-4 left-4">
              <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">Model</span>
            </div>
            <div className="absolute top-4 right-4">
              <span className="bg-pink-500/20 text-pink-400 text-xs px-2 py-1 rounded-md">Research</span>
            </div>
            
            <div className="mt-8">
              <h4 className="text-xl font-bold mb-3 group-hover:text-pink-400 transition-colors">
                Claude Sonnet 4
              </h4>
              <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                Anthropic's next-generation AI assistant with enhanced reasoning and multimodal capabilities
              </p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">Anthropic Research</span>
                <span className="text-pink-400 text-sm group-hover:text-pink-300 transition-colors">
                  Coming Soon →
                </span>
              </div>
            </div>
          </Link>
        </div>
      </main>
    </div>
  );
}