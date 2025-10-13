'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function HumansAndAI() {
  const { language } = useLanguage();

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Background effects */}
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
        </div>
        
        {/* Animated background particles */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/6 left-1/6 w-3 h-3 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full opacity-60 animate-pulse"></div>
          <div className="absolute top-1/4 right-1/5 w-2 h-2 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full opacity-50 animate-pulse delay-300"></div>
          <div className="absolute top-1/3 left-1/8 w-4 h-4 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full opacity-40 animate-pulse delay-700"></div>
          <div className="absolute bottom-1/4 right-1/6 w-2.5 h-2.5 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full opacity-55 animate-pulse delay-1000"></div>
        </div>
        
        <div className="relative container mx-auto px-6 pt-32 pb-12">
          <div className="text-center max-w-4xl mx-auto">
            {/* Title */}
            <div className="relative">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium mb-8 leading-tight">
                <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                  {language === 'en' ? 'Humans & AI' : '人类与AI'}
                </span>
              </h1>
              
              {/* Glow effect */}
              <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-medium leading-tight blur-sm">
                <span className="bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-cyan-400/20 bg-clip-text text-transparent">
                  {language === 'en' ? 'Humans & AI' : '人类与AI'}
                </span>
              </div>
            </div>
            
            {/* Subtitle */}
            <p className="text-lg md:text-xl text-slate-400 mb-8">
              {language === 'en' 
                ? 'Reading and writing essays on great books to deeply understand human nature and how we can guide human use of AI toward benefiting humanity.'
                : '阅读和撰写书籍评论，深入理解人性，以及如何引导人工智能的使用造福人类。'}
            </p>
          </div>
        </div>
      </section>

      {/* Books Grid Section */}
      <main className="bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 py-16">
        <div className="container mx-auto px-6">
          <div className="max-w-6xl mx-auto">
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {/* Lord of the Flies */}
              <Link 
                href="/humans-and-ai/lord-of-the-flies"
                className="group relative bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 hover:border-blue-500/50 hover:shadow-2xl hover:shadow-blue-500/10 transition-all duration-300"
              >
                <div className="absolute top-4 left-4">
                  <span className="bg-slate-600/50 text-slate-300 text-xs px-2 py-1 rounded-md">Classic Literature</span>
                </div>
                <div className="absolute top-4 right-4">
                  <span className="bg-blue-500/20 text-blue-400 text-xs px-2 py-1 rounded-md">New</span>
                </div>
                
                <div className="mt-12">
                  <h3 className="text-xl font-bold mb-3 group-hover:text-blue-400 transition-colors">
                    Lord of the Flies
                  </h3>
                  <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                    William Golding&apos;s masterpiece on human nature, civilization vs. savagery, and lessons for AI governance
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">William Golding (1954)</span>
                    <span className="text-blue-400 text-sm group-hover:text-blue-300 transition-colors">
                      Read Essay →
                    </span>
                  </div>
                </div>
              </Link>

              <div className="relative bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 border-dashed rounded-xl p-6 opacity-50">
                <div className="mt-12 text-center">
                  <div className="text-4xl mb-4">📖</div>
                  <p className="text-slate-500 text-sm">
                    {language === 'en' ? 'More books coming soon...' : '更多书籍即将推出...'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}

