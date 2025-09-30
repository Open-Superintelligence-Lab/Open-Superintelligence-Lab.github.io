'use client';

import Link from "next/link";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { useState, useEffect } from "react";

export default function ZeroToAIResearcherPage() {
  const [markdownContent, setMarkdownContent] = useState("");

  useEffect(() => {
    // Fetch the markdown content
    fetch('/zero-to-ai-researcher-content.md')
      .then(response => response.text())
      .then(content => setMarkdownContent(content))
      .catch(error => console.error('Error loading markdown content:', error));
  }, []);
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Blog Content */}
      <main className="container mx-auto px-6 pt-32 pb-12 max-w-4xl">
        <div className="mb-8">
          <Link href="/" className="text-slate-300 hover:text-white transition-colors">
            ← Back to Home
          </Link>
        </div>
        

        {/* Course Links */}
        <div className="mb-12">
          <div className="flex items-center gap-4 text-slate-400 text-sm mb-8">
            <span>Open Superintelligence Lab</span>
            <span>•</span>
            <span>September 2025</span>
            <span>•</span>
            <a 
              href="https://github.com/vukrosic/zero-to-ai-researcher/tree/main/_course" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-400 hover:text-blue-300 transition-colors"
            >
              View Course
            </a>
            <span>•</span>
            <a 
              href="https://github.com/vukrosic/zero-to-ai-researcher" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-400 hover:text-blue-300 transition-colors"
            >
              View Repository
            </a>
          </div>
        </div>

        {/* Markdown Content */}
        <article className="prose prose-invert max-w-none">
          {markdownContent ? (
            <MarkdownRenderer content={markdownContent} />
          ) : (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4"></div>
              <p className="text-slate-400">Loading content...</p>
            </div>
          )}
        </article>

        {/* Call to Action */}
        <section className="text-center mt-12">
          <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30 rounded-xl p-8">
            <h2 className="text-2xl font-bold mb-4 text-white">Ready to Start Your AI Research Journey?</h2>
            <p className="text-slate-300 mb-6">
              Join the open-source community and contribute to cutting-edge AI research
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a 
                href="https://github.com/vukrosic/zero-to-ai-researcher"
                target="_blank"
                rel="noopener noreferrer"
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-300 transform hover:scale-105"
              >
                Explore Repository
              </a>
              <Link 
                href="/about"
                className="px-6 py-3 border border-slate-600 text-slate-300 font-semibold rounded-lg hover:border-blue-500 hover:text-blue-400 transition-all duration-300"
              >
                Learn More About Us
              </Link>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
