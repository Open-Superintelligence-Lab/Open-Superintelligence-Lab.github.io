'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";
import { useEffect, useState } from "react";

export default function Home() {
  const { language } = useLanguage();
  const [isLocalhost, setIsLocalhost] = useState(false);

  useEffect(() => {
    setIsLocalhost(window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
  }, []);

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
                    {language === 'en' ? (
                      <>
                        <span className="bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent">Open</span>
                        <span className="mx-4 bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">Superintelligence</span>
                        <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">Lab</span>
                      </>
                    ) : (
                      <>
                        <span className="bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent">开放</span>
                        <span className="mx-4 bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">超级智能</span>
                        <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">实验室</span>
                      </>
                    )}
                  </h1>
                  
                  {/* Glow effect for the entire title */}
                  <div className="absolute inset-0 text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-medium leading-tight blur-sm">
                    {language === 'en' ? (
                      <>
                        <span className="bg-gradient-to-r from-green-400/20 via-emerald-400/20 to-teal-400/20 bg-clip-text text-transparent">Open</span>
                        <span className="mx-4 bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-cyan-400/20 bg-clip-text text-transparent">Superintelligence</span>
                        <span className="bg-gradient-to-r from-cyan-400/20 via-purple-400/20 to-blue-400/20 bg-clip-text text-transparent">Lab</span>
                      </>
                    ) : (
                      <>
                        <span className="bg-gradient-to-r from-green-400/20 via-emerald-400/20 to-teal-400/20 bg-clip-text text-transparent">开放</span>
                        <span className="mx-4 bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-cyan-400/20 bg-clip-text text-transparent">超级智能</span>
                        <span className="bg-gradient-to-r from-cyan-400/20 via-purple-400/20 to-blue-400/20 bg-clip-text text-transparent">实验室</span>
                      </>
                    )}
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
            <div className="flex flex-wrap justify-center gap-4 text-sm text-slate-400 mt-8 mb-12">
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

      {/* Main Projects Section */}
      <main id="research-projects" className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {/* Road to AI Researcher Project */}
            <Link 
              href="/blog/zero-to-ai-researcher"
              className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-green-500/50 hover:shadow-2xl hover:shadow-green-500/10 transition-all duration-300"
            >
              <div className="absolute top-4 left-4">
                <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">Learning Path</span>
              </div>
              <div className="absolute top-4 right-4">
                <span className="bg-green-500/20 text-green-400 text-xs px-2 py-1 rounded-md">Active</span>
              </div>
              
              <div className="mt-8">
                <h4 className="text-xl font-bold mb-3 group-hover:text-green-400 transition-colors">
                  Zero To AI Researcher - Full Course
                </h4>
                <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                  A comprehensive journey into becoming an AI researcher, covering everything from foundational concepts to cutting-edge research methodologies
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">Open Superintelligence Lab</span>
                  <span className="text-green-400 text-sm group-hover:text-green-300 transition-colors">
                    Read Blog Post →
                  </span>
                </div>
              </div>
            </Link>

            {/* DeepSeek Sparse Attention Project */}
            <Link 
              href="/blog/deepseek-sparse-attention"
              className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-blue-500/50 hover:shadow-2xl hover:shadow-blue-500/10 transition-all duration-300"
            >
              <div className="absolute top-4 left-4">
                <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">Research</span>
              </div>
              <div className="absolute top-4 right-4">
                <span className="bg-blue-500/20 text-blue-400 text-xs px-2 py-1 rounded-md">In Progress</span>
              </div>
              
              <div className="mt-8">
                <h4 className="text-xl font-bold mb-3 group-hover:text-blue-400 transition-colors">
                  DeepSeek Sparse Attention - DeepSeek-V3.2-Exp
                </h4>
                <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                  Advanced research on DeepSeek&apos;s innovative sparse attention mechanisms for efficient long-context processing and memory optimization
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">DeepSeek Research</span>
                  <span className="text-blue-400 text-sm group-hover:text-blue-300 transition-colors">
                    Learn More →
                  </span>
                </div>
              </div>
            </Link>
          </div>
        </div>
      </main>

      {/* Ideas Section - Only visible on localhost */}
      {isLocalhost && (
        <section className="bg-gradient-to-br from-slate-800 via-slate-700 to-slate-800 py-12">
          <div className="container mx-auto px-6">
            <div className="max-w-6xl mx-auto">
              <h2 className="text-2xl font-bold text-center mb-8 text-slate-300">
                {language === 'en' ? 'Research Ideas & Concepts' : '研究想法与概念'}
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* GLM4-MoE Project */}
                <a 
                  href="https://github.com/THUDM/GLM-4"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group relative bg-gradient-to-br from-slate-700/50 to-slate-600/50 backdrop-blur-sm border border-slate-500/50 rounded-lg p-4 hover:border-purple-500/50 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300"
                >
                  <div className="absolute top-2 left-2">
                    <span className="bg-slate-500/50 text-slate-200 text-xs px-2 py-1 rounded">Research</span>
                  </div>
                  <div className="absolute top-2 right-2">
                    <span className="bg-purple-500/20 text-purple-300 text-xs px-2 py-1 rounded">MoE</span>
                  </div>
                  
                  <div className="mt-6">
                    <h4 className="text-lg font-semibold mb-2 group-hover:text-purple-300 transition-colors">
                      GLM4-MoE
                    </h4>
                    <p className="text-gray-300 text-xs mb-3 leading-relaxed">
                      Advanced Mixture of Experts implementation with GLM4 architecture
                    </p>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">THUDM Research</span>
                      <span className="text-purple-300 text-xs group-hover:text-purple-200 transition-colors">
                        Explore →
                      </span>
                    </div>
                  </div>
                </a>

                {/* DeepSeek Attention + GLM4-MoE Project */}
                <a 
                  href="https://github.com/Open-Superintelligence-Lab/deepseek-attention-glm4-moe"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group relative bg-gradient-to-br from-slate-700/50 to-slate-600/50 backdrop-blur-sm border border-slate-500/50 rounded-lg p-4 hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10 transition-all duration-300"
                >
                  <div className="absolute top-2 left-2">
                    <span className="bg-slate-500/50 text-slate-200 text-xs px-2 py-1 rounded">Research</span>
                  </div>
                  <div className="absolute top-2 right-2">
                    <span className="bg-cyan-500/20 text-cyan-300 text-xs px-2 py-1 rounded">Hybrid</span>
                  </div>
                  
                  <div className="mt-6">
                    <h4 className="text-lg font-semibold mb-2 group-hover:text-cyan-300 transition-colors">
                      DeepSeek Attention + GLM4-MoE
                    </h4>
                    <p className="text-gray-300 text-xs mb-3 leading-relaxed">
                      Combination of DeepSeek&apos;s sparse attention with GLM4&apos;s MoE
                    </p>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">Open Superintelligence Lab</span>
                      <span className="text-cyan-300 text-xs group-hover:text-cyan-200 transition-colors">
                        Learn More →
                      </span>
                    </div>
                  </div>
                </a>

                {/* SLA Sparse-Linear Attention Project */}
                <Link 
                  href="/blog/sla-sparse-linear-attention"
                  className="group relative bg-gradient-to-br from-slate-700/50 to-slate-600/50 backdrop-blur-sm border border-slate-500/50 rounded-lg p-4 hover:border-orange-500/50 hover:shadow-lg hover:shadow-orange-500/10 transition-all duration-300"
                >
                  <div className="absolute top-2 left-2">
                    <span className="bg-slate-500/50 text-slate-200 text-xs px-2 py-1 rounded">Research</span>
                  </div>
                  <div className="absolute top-2 right-2">
                    <span className="bg-orange-500/20 text-orange-300 text-xs px-2 py-1 rounded">New</span>
                  </div>
                  
                  <div className="mt-6">
                    <h4 className="text-lg font-semibold mb-2 group-hover:text-orange-300 transition-colors">
                      SLA: Sparse-Linear Attention
                    </h4>
                    <p className="text-gray-300 text-xs mb-3 leading-relaxed">
                      20x speedup with minimal quality loss in diffusion transformers
                    </p>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">Tsinghua University</span>
                      <span className="text-orange-300 text-xs group-hover:text-orange-200 transition-colors">
                        Learn More →
                      </span>
                    </div>
                  </div>
                </Link>

                {/* SDLM Sequential Diffusion Language Model Project */}
                <Link 
                  href="/blog/sdlm-sequential-diffusion-language-model"
                  className="group relative bg-gradient-to-br from-slate-700/50 to-slate-600/50 backdrop-blur-sm border border-slate-500/50 rounded-lg p-4 hover:border-purple-500/50 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300"
                >
                  <div className="absolute top-2 left-2">
                    <span className="bg-slate-500/50 text-slate-200 text-xs px-2 py-1 rounded">Research</span>
                  </div>
                  <div className="absolute top-2 right-2">
                    <span className="bg-purple-500/20 text-purple-300 text-xs px-2 py-1 rounded">Latest</span>
                  </div>
                  
                  <div className="mt-6">
                    <h4 className="text-lg font-semibold mb-2 group-hover:text-purple-300 transition-colors">
                      SDLM: Sequential Diffusion Language Model
                    </h4>
                    <p className="text-gray-300 text-xs mb-3 leading-relaxed">
                      Adaptive generation length with 2x speedup and KV-cache compatibility
                    </p>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">OpenGVLab</span>
                      <span className="text-purple-300 text-xs group-hover:text-purple-200 transition-colors">
                        Learn More →
                      </span>
                    </div>
                  </div>
                </Link>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Localhost Todos Section - Only visible on localhost */}
      {isLocalhost && (
        <section className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
          <div className="container mx-auto px-6">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-2xl font-bold text-center mb-8 text-slate-300">
                {language === 'en' ? 'Development Todos' : '开发待办事项'}
              </h2>
              
              {/* Dream-Coder 7B Todo */}
              <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 mb-6">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 mt-1">
                    <div className="w-4 h-4 border-2 border-blue-400 rounded-full bg-blue-400/20"></div>
                  </div>
                  <div className="flex-grow">
                    <h3 className="text-lg font-semibold text-blue-400 mb-2">
                      Dream-Coder 7B Exploration
                    </h3>
                    <p className="text-slate-300 text-sm mb-4">
                      Explore Dream-Coder 7B - diffusion LLM for code generation with 21.4% pass@1 on LiveCodeBench
                    </p>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Review GitHub repository: https://github.com/DreamLM/Dream-Coder</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Understand diffusion LLM architecture for code</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Test code generation capabilities</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Experiment with flexible generation patterns</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Install dependencies: transformers==4.46.2 and torch==2.5.1</span>
                      </div>
                    </div>
                    <div className="mt-4 pt-4 border-t border-slate-600/50">
                      <div className="flex items-center gap-2 text-xs text-slate-500">
                        <div className="w-2 h-2 bg-blue-400/30 rounded-full"></div>
                        <span>Features: Emergent any-order generation, variable-length infilling, open-source trained</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* NVFP4 Research Todo */}
              <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 mb-6">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 mt-1">
                    <div className="w-4 h-4 border-2 border-green-400 rounded-full bg-green-400/20"></div>
                  </div>
                  <div className="flex-grow">
                    <h3 className="text-lg font-semibold text-green-400 mb-2">
                      NVFP4 LLM Pretraining Research
                    </h3>
                    <p className="text-slate-300 text-sm mb-4">
                      Research NVIDIA's NVFP4 (4-bit floating point) training methodology - 2-3x performance boost with 50% memory reduction
                    </p>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Study paper: https://arxiv.org/pdf/2509.25149</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>https://github.com/NVIDIA/TransformerEngine/pull/2177/files - Code</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Learn Random Hadamard Transforms (RHT) for outlier bounding</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Study two-dimensional quantization scheme</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Research stochastic rounding for unbiased gradients</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Set up experimental environment for FP4 training</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Study TransformerEngine NVFP4 implementation (PR #2177)</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Test NVFP4 support with fusible operations</span>
                      </div>
                    </div>
                    <div className="mt-4 pt-4 border-t border-slate-600/50">
                      <div className="flex items-center gap-2 text-xs text-slate-500">
                        <div className="w-2 h-2 bg-green-400/30 rounded-full"></div>
                        <span>Results: 12B model on 10T tokens, MMLU-pro 62.58% (vs 62.62% FP8) - first successful 4-bit billion-parameter training</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* MobileLLM-R1 Research Todo */}
              <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 mb-6">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 mt-1">
                    <div className="w-4 h-4 border-2 border-purple-400 rounded-full bg-purple-400/20"></div>
                  </div>
                  <div className="flex-grow">
                    <h3 className="text-lg font-semibold text-purple-400 mb-2">
                      MobileLLM-R1 Sub-Billion Reasoning Research
                    </h3>
                    <p className="text-slate-300 text-sm mb-4">
                      Research Meta's MobileLLM-R1 - sub-billion parameter reasoning models with strong capabilities using only ~2T high-quality tokens
                    </p>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Study paper: https://arxiv.org/pdf/2509.24945</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Understand data curation and resampling techniques (~2T tokens)</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Learn benchmark-free, self-evolving data optimization</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Study data-model co-evolution strategy for mid-training</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Analyze training recipe: 4.2T tokens from resampled ~2T tokens</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <div className="w-2 h-2 border border-slate-500 rounded-full"></div>
                        <span>Test MobileLLM-R1-950M model capabilities</span>
                      </div>
                    </div>
                    <div className="mt-4 pt-4 border-t border-slate-600/50">
                      <div className="flex items-center gap-2 text-xs text-slate-500">
                        <div className="w-2 h-2 bg-purple-400/30 rounded-full"></div>
                        <span>Results: AIME 15.5 vs OLMo-2-1.48B (0.6), matches Qwen3-0.6B with only 11.7% of tokens (4.2T vs 36T)</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}
    </>
  );
}