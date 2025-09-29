'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { useState, useEffect } from "react";

export default function DeepSeekV32ExpPage() {
  const { language } = useLanguage();
  const [markdownContent, setMarkdownContent] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch the markdown content from the external file
    fetch('/content-research-deepseek-v3-2-exp.md')
      .then(res => res.text())
      .then(content => {
        setMarkdownContent(content);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading markdown:', error);
        setLoading(false);
      });
  }, []);

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-orange-600/20 via-yellow-600/20 to-orange-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-orange-500/5 to-transparent"></div>
        </div>
        
        <div className="relative container mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-orange-400 to-yellow-400 bg-clip-text text-transparent">
              V3.2-Exp Architecture
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              {language === 'en' 
                ? "Open source research on DeepSeek Sparse Attention (DSA) and long-context efficiency improvements"
                : "DeepSeek稀疏注意力(DSA)和长上下文效率改进的开源研究"
              }
            </p>
            <div className="flex justify-center gap-4">
              <span className="bg-orange-500/20 text-orange-400 text-sm px-3 py-1 rounded-md">Open Source</span>
              <span className="bg-yellow-500/20 text-yellow-400 text-sm px-3 py-1 rounded-md">Research Article</span>
            </div>
          </div>
        </div>
      </section>

      <main className="container mx-auto px-6 py-12">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <Link 
              href="/research/deepseek" 
              className="text-gray-400 hover:text-white transition-colors inline-flex items-center gap-2"
            >
              <span>←</span>
              <span>{language === 'en' ? 'Back to DeepSeek Research' : '返回DeepSeek研究'}</span>
            </Link>
          </div>

          {/* Markdown Content */}
          {loading ? (
            <div className="flex justify-center items-center py-20">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-400"></div>
            </div>
          ) : (
            <article className="bg-slate-900/50 border border-gray-800 rounded-lg p-8 md:p-12">
              <MarkdownRenderer content={markdownContent} />
            </article>
          )}

          {/* Back Navigation */}
          <div className="text-center mt-12">
            <Link 
              href="/research/deepseek" 
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-slate-700/50 to-slate-600/50 border border-slate-500/50 rounded-lg hover:border-orange-500/50 hover:bg-slate-600/50 transition-all duration-200"
            >
              <span>←</span>
              <span>{language === 'en' ? 'Back to DeepSeek Research' : '返回DeepSeek研究'}</span>
            </Link>
          </div>
        </div>
      </main>
    </>
  );
}