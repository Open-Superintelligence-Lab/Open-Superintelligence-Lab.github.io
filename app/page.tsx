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
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
        </div>
        
        <div className="relative container mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              {t.title}
            </h1>
            <h2 className="text-xl md:text-2xl text-gray-300 mb-8">
              {t.subtitle}
            </h2>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto">
              {t.description}
            </p>
          </div>
        </div>
      </section>

      {/* Projects Section */}
      <main className="container mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h3 className="text-3xl font-bold mb-4">{t.projects}</h3>
          <p className="text-gray-400">Explore our cutting-edge AI research projects</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
          {/* DeepSeek Project */}
          <Link 
            href="/research/deepseek-v3-2-exp"
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