'use client';

import Link from "next/link";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { useEffect, useState } from "react";

interface HeroData {
  title: string;
  subtitle: string;
  tags: string[];
}

export default function NVFP4Project() {
  const [markdownContent, setMarkdownContent] = useState<string>('');
  const [heroData, setHeroData] = useState<HeroData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [copySuccess, setCopySuccess] = useState(false);

  useEffect(() => {
    const fetchMarkdownContent = async () => {
      try {
        const response = await fetch(`/content/pretrain-llm-with-nvfp4/pretrain-llms-with-fp4-content.md`);
        const content = await response.text();

        // Parse frontmatter
        const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
        if (frontmatterMatch) {
          const frontmatterContent = frontmatterMatch[1];
          const markdownBody = frontmatterMatch[2];

          // Parse YAML-like frontmatter (simple parsing for our use case)
          const heroData: HeroData = {
            title: "NVIDIA's 4-Bit Revolution",
            subtitle: "⚡ NVFP4: 2-3x Faster Training, 50% Less Memory",
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
  }, []);

  const handleCopyArticle = async () => {
    try {
      // Get the raw markdown content without frontmatter
      const response = await fetch(`/content/pretrain-llm-with-nvfp4/pretrain-llms-with-fp4-content.md`);
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
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-400 mx-auto mb-4"></div>
          <p className="text-slate-400">Loading article content...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Hero Section - Reduced */}
      <section className="relative overflow-hidden bg-gradient-to-br from-background via-card/50 to-background border-b border-border/30">
        {/* Simplified background - single subtle gradient */}
        <div className="absolute inset-0 bg-gradient-to-r from-gradient-accent-1/5 via-gradient-accent-2/5 to-gradient-accent-3/5"></div>

        <div className="relative container mx-auto px-6 pt-16 pb-6">
          <div className="text-left max-w-4xl mx-auto">
            <div className="relative">
              <h1 className="text-3xl md:text-4xl lg:text-5xl font-semibold mb-3 leading-tight">
                <span className="text-foreground">
                  {heroData?.title || "NVIDIA's 4-Bit Revolution"}
                </span>
              </h1>
              <div className="text-base md:text-lg text-muted-foreground mb-4">
                {heroData?.subtitle || "⚡ NVFP4: 2-3x Faster Training, 50% Less Memory"}
              </div>

              {/* Tags */}
              {heroData?.tags && heroData.tags.length > 0 && (
                <div className="flex items-center gap-3 text-sm text-muted-foreground mb-4">
                  {heroData.tags.map((tag, index) => (
                    <span key={index} className="flex items-center gap-2">
                      {index > 0 && <span className="text-muted-foreground/40">•</span>}
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
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 ${copySuccess
                          ? 'text-green-400 bg-green-400/10 border border-green-400/20'
                          : 'text-slate-400 hover:text-green-400 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-500/50'
                        }`}
                    >
                      {copySuccess ? (
                        <>
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          <span className="text-sm font-medium">
                            Copied!
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
                      Perfect for pasting into AI chatbots for self-studying! 🤖
                      {/* Tooltip arrow */}
                      <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-800"></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Article Body */}
              <div className="px-8 sm:px-12 pb-20">
                <div className="mb-8">
                  <div className="relative" style={{ paddingTop: '56.25%' }}>
                    <iframe
                      className="absolute top-0 left-0 w-full h-full rounded-lg shadow-2xl"
                      src="https://www.youtube.com/embed/Rk4APYLK0VY"
                      title="YouTube video player"
                      frameBorder="0"
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                    ></iframe>
                  </div>
                  <div className="mt-4 text-center">
                    <p className="text-slate-400 text-sm mb-2">Also available on:</p>
                    <a
                      href="https://www.bilibili.com/video/BV1WJxtzeEo4/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 px-4 py-2 bg-blue-800 hover:bg-blue-900 text-white rounded-lg transition-colors text-sm font-medium"
                    >
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M23.182 5.656C22.874 4.67 22.04 3.836 21.054 3.528 19.29 3 12 3 12 3S4.71 3 2.946 3.528C1.96 3.836 1.126 4.67.818 5.656.29 7.42.29 12 .29 12S.29 16.58.818 18.344C1.126 19.33 1.96 20.164 2.946 20.472C4.71 21 12 21 12 21S19.29 21 21.054 20.472C22.04 20.164 22.874 19.33 23.182 18.344C23.71 16.58 23.71 12 23.71 12S23.71 7.42 23.182 5.656ZM9.955 15.465V8.535L15.818 12L9.955 15.465Z" />
                      </svg>
                      Watch on Bilibili
                    </a>
                  </div>
                </div>
                <div className="prose prose-lg prose-invert max-w-none">
                  <MarkdownRenderer content={markdownContent} />
                </div>
              </div>

              {/* Article Footer */}
              <div className="bg-gradient-to-r from-green-600/5 via-emerald-600/5 to-teal-600/5 border-t border-white/10 px-8 sm:px-12 py-8">
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
                        className={`flex items-center justify-center p-2 rounded-lg transition-all duration-300 ${copySuccess
                            ? 'text-green-400'
                            : 'text-slate-400 hover:text-green-400'
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
                        Perfect for pasting into AI chatbots for self-studying! 🤖
                        {/* Tooltip arrow */}
                        <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-800"></div>
                      </div>
                    </div>

                    <a href="https://x.com/intent/tweet?text=Check%20out%20this%20article%20about%20NVIDIA%27s%20NVFP4%20-%20how%20to%20train%20LLMs%20with%204-bit%20precision%20%F0%9F%9A%80%0A%0A2-3x%20faster%20training%2C%2050%25%20less%20memory.%20Covers%20architecture%2C%20implementation%2C%20and%20benchmarks.%0A%0A%23AI%20%23MachineLearning%20%23DeepLearning&url=https://opensuperintelligencelab.com/blog/pretrain-llm-with-nvfp4/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-slate-400 hover:text-green-400 transition-colors">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                      </svg>
                    </a>
                    <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://opensuperintelligencelab.com/blog/pretrain-llm-with-nvfp4/&title=Check%20out%20this%20article%20about%20NVIDIA%20NVFP4%3A%204-Bit%20LLM%20Training&summary=Check%20out%20this%20technical%20article%20about%20NVIDIA%27s%20NVFP4%20for%204-bit%20LLM%20training.%20Achieves%202-3x%20faster%20training%20speeds%20and%2050%25%20memory%20reduction.%20Includes%20architecture%20details%2C%20implementation%20guide%2C%20and%20benchmarks."
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-slate-400 hover:text-green-400 transition-colors">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
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
                className="group flex items-center gap-2 px-6 py-3 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-green-500/50 text-slate-300 hover:text-green-400 font-medium rounded-xl transition-all duration-300"
              >
                <svg className="w-5 h-5 group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Back to Home
              </Link>

              <div className="flex items-center gap-2 text-sm text-slate-500">
                <span className="hidden sm:inline">Scroll to</span>
                <button
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                  className="flex items-center gap-1 px-4 py-2 hover:text-green-400 transition-colors"
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
