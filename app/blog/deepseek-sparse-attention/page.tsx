'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { useEffect, useState } from "react";

export default function DeepSeekProject() {
  const { language } = useLanguage();
  const [markdownContent, setMarkdownContent] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchMarkdownContent = async () => {
      try {
        const response = await fetch('/deepseek-sparse-attention-content.md');
        const content = await response.text();
        setMarkdownContent(content);
      } catch (error) {
        console.error('Failed to fetch markdown content:', error);
        setMarkdownContent('# Error loading content\n\nFailed to load the article content.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchMarkdownContent();
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4"></div>
          <p className="text-slate-400">Loading article content...</p>
        </div>
      </div>
    );
  }

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
        
        <div className="relative container mx-auto px-6 pt-32 pb-24">
          <div className="text-center max-w-4xl mx-auto">
            <div className="relative">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium mb-8 leading-tight">
                <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                  {language === 'en' ? 'DeepSeek\'s Attention Revolution' : 'DeepSeek 的注意力革命'}
                </span>
              </h1>
              <div className="text-lg md:text-xl text-slate-400 mb-4">
                {language === 'en' 
                  ? '⚡ From O(L²) to O(Lk) - The Lightning Indexer Breakthrough'
                  : '⚡ 从 O(L²) 到 O(Lk) - 闪电索引器突破'
                }
              </div>
              
              {/* Glow effect for the title */}
              <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-medium leading-tight blur-sm">
                <span className="bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-cyan-400/20 bg-clip-text text-transparent">
                  {language === 'en' ? 'DeepSeek\'s Attention Revolution' : 'DeepSeek 的注意力革命'}
                </span>
              </div>
            </div>
            
            <p className="text-xl text-slate-300 mb-12 leading-relaxed">
              {language === 'en' 
                ? 'A deep dive into sparse attention and the Lightning Indexer - DeepSeek-V3.2-Exp'
                : '深入探讨稀疏注意力和闪电索引器 - DeepSeek-V3.2-Exp'
              }
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 min-h-screen py-20">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          {/* Article Container */}
          <article className="max-w-4xl mx-auto">
            {/* Content Card */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl shadow-2xl overflow-hidden">
              {/* Article Header */}
              <div className="bg-gradient-to-r from-blue-600/10 via-purple-600/10 to-cyan-600/10 border-b border-white/10 px-8 sm:px-12 py-8">
                <div className="flex items-center gap-3 text-sm text-slate-400 mb-4">
                  <span className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Technical Deep Dive
                  </span>
                  <span className="text-slate-600">•</span>
                  <span className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Research Article
                  </span>
                </div>
                <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent mb-4">
                  DeepSeek Sparse Attention
                </h1>
                <p className="text-slate-400 text-lg leading-relaxed">
                  A comprehensive exploration of the Lightning Indexer and Multi-Head Latent Attention architecture
                </p>
              </div>

              {/* Article Body */}
              <div className="px-8 sm:px-12 py-12">
                <div className="prose prose-lg prose-invert max-w-none">
                  <MarkdownRenderer content={markdownContent} />
                </div>
              </div>

              {/* Article Footer */}
              <div className="bg-gradient-to-r from-blue-600/5 via-purple-600/5 to-cyan-600/5 border-t border-white/10 px-8 sm:px-12 py-8">
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                  <div className="flex items-center gap-3 text-sm text-slate-400">
                    <span className="flex items-center gap-2">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                      Open Superintelligence Lab
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Share</span>
                    <a href="https://twitter.com/intent/tweet?text=DeepSeek%20Sparse%20Attention%20-%20Lightning%20Indexer" 
                       target="_blank" 
                       rel="noopener noreferrer"
                       className="text-slate-400 hover:text-blue-400 transition-colors">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                      </svg>
                    </a>
                    <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://opensuperintelligencelab.com" 
                       target="_blank" 
                       rel="noopener noreferrer"
                       className="text-slate-400 hover:text-blue-400 transition-colors">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                      </svg>
                    </a>
                  </div>
                </div>
              </div>
          </div>

            {/* Navigation */}
            <div className="mt-12 flex flex-col sm:flex-row items-center justify-between gap-4">
            <Link 
              href="/"
                className="group flex items-center gap-2 px-6 py-3 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-500/50 text-slate-300 hover:text-blue-400 font-medium rounded-xl transition-all duration-300"
            >
                <svg className="w-5 h-5 group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              {language === 'en' ? 'Back to Home' : '返回首页'}
            </Link>
              
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <span className="hidden sm:inline">Scroll to</span>
                <button 
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                  className="flex items-center gap-1 px-4 py-2 hover:text-blue-400 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                  </svg>
                  Top
                </button>
              </div>
          </div>
          </article>
        </div>
      </main>
    </>
  );
}
