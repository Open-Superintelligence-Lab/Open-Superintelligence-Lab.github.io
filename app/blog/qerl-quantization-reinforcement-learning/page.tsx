'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { useEffect, useState } from "react";

interface HeroData {
  title: string;
  subtitle: string;
  tags: string[];
}

export default function QeRLProject() {
  const { language } = useLanguage();
  const [markdownContent, setMarkdownContent] = useState<string>('');
  const [heroData, setHeroData] = useState<HeroData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [copySuccess, setCopySuccess] = useState(false);

  useEffect(() => {
    const fetchMarkdownContent = async () => {
      try {
        const filename = language === 'zh' ? 'qerl-content-zh.md' : 'qerl-content.md';
        const response = await fetch(`/content/qerl-quantization-reinforcement-learning/${filename}`);
        const content = await response.text();
        
        // Parse frontmatter
        const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
        if (frontmatterMatch) {
          const frontmatterContent = frontmatterMatch[1];
          const markdownBody = frontmatterMatch[2];
          
          // Parse YAML-like frontmatter (simple parsing for our use case)
          const heroData: HeroData = {
            title: "QeRL: Beyond Efficiency",
            subtitle: "Quantization-enhanced Reinforcement Learning for LLMs",
            tags: ["⏱️ Technical Deep Dive", "📄 Research Article"]
          };
          
          // Extract values from frontmatter
          const lines = frontmatterContent.split('\n');
          let currentKey = '';
          let currentArray: string[] = [];
          
          for (const line of lines) {
            const trimmedLine = line.trim();
            if (trimmedLine.startsWith('hero:')) continue;
            
            if (trimmedLine.includes(':')) {
              const [key, ...valueParts] = trimmedLine.split(':');
              const value = valueParts.join(':').trim().replace(/^["']|["']$/g, '');
              
              switch (key.trim()) {
                case 'title':
                  heroData.title = value;
                  break;
                case 'subtitle':
                  heroData.subtitle = value;
                  break;
                case 'tags':
                  currentKey = 'tags';
                  currentArray = [];
                  break;
              }
            } else if (trimmedLine.startsWith('- ')) {
              if (currentKey === 'tags') {
                const tagValue = trimmedLine.substring(2).replace(/^["']|["']$/g, '');
                currentArray.push(tagValue);
              }
            } else if (trimmedLine === '' && currentArray.length > 0) {
              if (currentKey === 'tags') {
                heroData.tags = currentArray;
                currentArray = [];
                currentKey = '';
              }
            }
          }
          
          // Handle final array
          if (currentArray.length > 0 && currentKey === 'tags') {
            heroData.tags = currentArray;
          }
          
          setHeroData(heroData);
          setMarkdownContent(markdownBody);
        } else {
          // Fallback if no frontmatter
          setMarkdownContent(content);
        }
      } catch (error) {
        console.error('Failed to fetch markdown content:', error);
        setMarkdownContent('# Error loading content\n\nFailed to load the article content.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchMarkdownContent();
  }, [language]);

  const handleCopyArticle = async () => {
    try {
      // Get the raw markdown content without frontmatter
      const filename = language === 'zh' ? 'qerl-content-zh.md' : 'qerl-content.md';
      const response = await fetch(`/content/qerl-quantization-reinforcement-learning/${filename}`);
      const content = await response.text();
      
      // Remove frontmatter if present
      let contentWithoutFrontmatter = content.replace(/^---\n[\s\S]*?\n---\n/, '');
      
      // Remove image paths (markdown image syntax: ![alt text](image-path))
      contentWithoutFrontmatter = contentWithoutFrontmatter.replace(/!\[.*?\]\(.*?\)/g, '');
      
      await navigator.clipboard.writeText(contentWithoutFrontmatter);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (error) {
      console.error('Failed to copy article:', error);
    }
  };

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
        
        <div className="relative container mx-auto px-6 pt-32 pb-12">
          <div className="text-center max-w-4xl mx-auto">
            <div className="relative">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium mb-8 leading-tight">
                <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                  {heroData?.title || 'QeRL: Beyond Efficiency'}
                </span>
              </h1>
              <div className="text-lg md:text-xl text-slate-400 mb-8">
                {heroData?.subtitle || 'Quantization-enhanced Reinforcement Learning for LLMs'}
              </div>
              
              {/* Tags */}
              {heroData?.tags && heroData.tags.length > 0 && (
                <div className="flex items-center justify-center gap-3 text-sm text-slate-400 mb-8">
                  {heroData.tags.map((tag, index) => (
                    <span key={index} className="flex items-center gap-2">
                      {index > 0 && <span className="text-slate-600">•</span>}
                      <span className="flex items-center gap-2">
                        {tag.includes('⏱️') && (
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        )}
                        {tag.includes('📄') && (
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        )}
                        {tag.replace(/[⏱️📄]/g, '').trim()}
                      </span>
                    </span>
                  ))}
                </div>
              )}

              {/* Links to Paper and GitHub */}
              <div className="relative z-10 flex flex-wrap items-center justify-center gap-4 mb-8">
                <a 
                  href="https://arxiv.org/pdf/2510.11696"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600/20 to-purple-600/20 hover:from-blue-600/30 hover:to-purple-600/30 border border-blue-500/30 hover:border-blue-400/50 text-blue-300 hover:text-blue-200 font-medium rounded-xl transition-all duration-300 shadow-lg hover:shadow-blue-500/25"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span>Read Paper</span>
                  <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
                <a 
                  href="https://github.com/NVlabs/QeRL"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600/20 to-pink-600/20 hover:from-purple-600/30 hover:to-pink-600/30 border border-purple-500/30 hover:border-purple-400/50 text-purple-300 hover:text-purple-200 font-medium rounded-xl transition-all duration-300 shadow-lg hover:shadow-purple-500/25"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                  </svg>
                  <span>View Code</span>
                  <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              </div>
              
              {/* Glow effect for the title */}
              <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-medium leading-tight blur-sm pointer-events-none">
                <span className="bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-cyan-400/20 bg-clip-text text-transparent">
                  {heroData?.title || 'QeRL: Beyond Efficiency'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 min-h-screen">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 pt-8 pb-16">
          {/* Article Container */}
          <article className="max-w-4xl mx-auto">
            {/* Content Card */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl shadow-2xl">
              {/* Copy Button at Top */}
              <div className="px-8 sm:px-12 pt-8 pb-4">
                <div className="flex justify-end">
                  <div className="relative inline-block group">
                    <button
                      onClick={handleCopyArticle}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 ${
                        copySuccess
                          ? 'text-green-400 bg-green-400/10 border border-green-400/20'
                          : 'text-slate-400 hover:text-blue-400 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-500/50'
                      }`}
                    >
                      {copySuccess ? (
                        <>
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          <span className="text-sm font-medium">
                            {language === 'en' ? 'Copied!' : '已复制!'}
                          </span>
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                        </>
                      )}
                    </button>
                    
                    {/* Tooltip */}
                    <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-slate-800 text-white text-sm rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-10 border border-slate-600">
                      {language === 'en' 
                        ? 'Perfect for pasting into AI chatbots for self-studying! 🤖' 
                        : '非常适合粘贴到AI聊天机器人进行自学！🤖'
                      }
                      {/* Tooltip arrow */}
                      <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-800"></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Article Body */}
              <div className="px-8 sm:px-12 pb-20">
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
                    
                    {/* Copy Article Button */}
                    <div className="relative inline-block group">
                      <button
                        onClick={handleCopyArticle}
                        className={`flex items-center justify-center p-2 rounded-lg transition-all duration-300 ${
                          copySuccess
                            ? 'text-green-400'
                            : 'text-slate-400 hover:text-blue-400'
                        }`}
                      >
                        {copySuccess ? (
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        ) : (
                          <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                        )}
                      </button>
                      
                      {/* Tooltip */}
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-slate-800 text-white text-sm rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-10 border border-slate-600">
                        {language === 'en' 
                          ? 'Perfect for pasting into AI chatbots for self-studying! 🤖' 
                          : '非常适合粘贴到AI聊天机器人进行自学！🤖'
                        }
                        {/* Tooltip arrow */}
                        <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-800"></div>
                      </div>
                    </div>
                    
                    <a href="https://twitter.com/intent/tweet?text=Check%20out%20QeRL%20-%20Quantization-enhanced%20Reinforcement%20Learning%20for%20LLMs&url=https://opensuperintelligencelab.com/blog/qerl-quantization-reinforcement-learning/" 
                       target="_blank" 
                       rel="noopener noreferrer"
                       className="text-slate-400 hover:text-blue-400 transition-colors">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                      </svg>
                    </a>
                    <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://opensuperintelligencelab.com/blog/qerl-quantization-reinforcement-learning/" 
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


