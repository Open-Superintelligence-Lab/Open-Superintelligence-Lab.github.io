'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function BlueberryLLMPage() {
  const { language } = useLanguage();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700/50">
        <div className="container mx-auto px-6 py-4">
          <Link href="/" className="text-slate-300 hover:text-white transition-colors">
            ← Back to Home
          </Link>
        </div>
      </header>

      {/* Blog Content */}
      <main className="container mx-auto px-6 py-12 max-w-4xl">
        <article className="prose prose-invert max-w-none">
          {/* Title */}
          <header className="mb-12">
            <h1 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
              Zero To AI Researcher - Full Course
            </h1>
            <div className="flex items-center gap-4 text-slate-400 text-sm">
              <span>Open Superintelligence Lab</span>
              <span>•</span>
              <span>January 2025</span>
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
          </header>

          {/* Video Placeholder */}
          <div className="mb-12">
            <div className="aspect-video bg-slate-800/50 rounded-xl border border-slate-600/50 flex items-center justify-center">
              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z"/>
                  </svg>
                </div>
                <p className="text-slate-400">Course Introduction Video</p>
                <p className="text-slate-500 text-sm">Coming Soon</p>
              </div>
            </div>
          </div>

          {/* Introduction */}
          <section className="mb-12">
            <p className="text-lg text-slate-300 leading-relaxed mb-6">
              The <strong>Blueberry LLM Research Course</strong> is a comprehensive educational journey that takes you from zero to AI researcher, 
              focusing on cutting-edge transformer architectures including Mixture of Experts (MoE) models inspired by DeepSeek and GLM4.
            </p>
            <p className="text-slate-300 leading-relaxed">
              This course provides hands-on experience with advanced AI research methodologies, from foundational concepts to implementing 
              state-of-the-art architectures. Whether you're a beginner or looking to deepen your understanding of modern LLM research, 
              this course offers a structured path to becoming an AI researcher.
            </p>
          </section>

          {/* Key Questions */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-6 text-white">Key Questions This Course Answers</h2>
            <div className="space-y-6">
              <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-3 text-blue-400">1. How do Mixture of Experts (MoE) models work and why are they crucial for scaling LLMs?</h3>
                <p className="text-slate-300">
                  Learn the fundamentals of MoE architecture, expert routing mechanisms, and load balancing strategies that enable 
                  efficient scaling of language models while maintaining computational efficiency.
                </p>
              </div>
              
              <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-3 text-purple-400">2. What makes DeepSeek's attention mechanisms innovative and how can they be combined with MoE?</h3>
                <p className="text-slate-300">
                  Explore DeepSeek's advanced attention mechanisms, including sparse attention patterns, RoPE scaling, and LoRA-style projections, 
                  and understand how to integrate them with MoE architectures for optimal performance.
                </p>
              </div>
              
              <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-3 text-cyan-400">3. How do you conduct systematic AI research and evaluate different architectures?</h3>
                <p className="text-slate-300">
                  Master the art of AI research through structured experimentation, ablation studies, benchmarking methodologies, 
                  and learn how to design and execute comprehensive research projects in the field of large language models.
                </p>
              </div>
            </div>
          </section>

          {/* Course Structure */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-6 text-white">Course Structure</h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-4 text-green-400">Foundation Modules</h3>
                <ul className="space-y-2 text-slate-300">
                  <li>• Python fundamentals for AI research</li>
                  <li>• Mathematical foundations (linear algebra, calculus)</li>
                  <li>• PyTorch basics and tensor operations</li>
                  <li>• Neural network fundamentals</li>
                  <li>• Activation functions and optimization</li>
                </ul>
              </div>
              
              <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-4 text-blue-400">Advanced Topics</h3>
                <ul className="space-y-2 text-slate-300">
                  <li>• Attention mechanisms and transformers</li>
                  <li>• DeepSeek attention implementation</li>
                  <li>• GLM4 MoE architecture</li>
                  <li>• Hybrid model combinations</li>
                  <li>• Research methodology and evaluation</li>
                </ul>
              </div>
            </div>
          </section>

          {/* Key Features */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-6 text-white">What You'll Build</h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6 text-center">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h3 className="font-semibold mb-2 text-white">MoE Models</h3>
                <p className="text-slate-400 text-sm">Implement efficient Mixture of Experts architectures with expert routing and load balancing</p>
              </div>
              
              <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6 text-center">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="font-semibold mb-2 text-white">DeepSeek Attention</h3>
                <p className="text-slate-400 text-sm">Build advanced attention mechanisms with RoPE scaling and sparse patterns</p>
              </div>
              
              <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6 text-center">
                <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h3 className="font-semibold mb-2 text-white">Research Framework</h3>
                <p className="text-slate-400 text-sm">Develop systematic experimentation and evaluation methodologies</p>
              </div>
            </div>
          </section>

          {/* Experiments */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-6 text-white">Hands-On Experiments</h2>
            <div className="space-y-4">
              <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-2 text-yellow-400">Experiment 1: Architecture Ablation Study</h3>
                <p className="text-slate-300 text-sm mb-3">
                  Compare 5 different model variants (baseline, MLP, attention+MLP, MoE, attention+MoE) to understand the impact of each component.
                </p>
                <span className="inline-block bg-slate-700/50 text-slate-300 text-xs px-2 py-1 rounded">HellaSwag Benchmark</span>
              </div>
              
              <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-2 text-orange-400">Experiment 2: Learning Rate Optimization</h3>
                <p className="text-slate-300 text-sm mb-3">
                  Systematic exploration of optimal learning rates for DeepSeek attention + MLP combinations.
                </p>
                <span className="inline-block bg-slate-700/50 text-slate-300 text-xs px-2 py-1 rounded">Grid Search</span>
              </div>
              
              <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-2 text-red-400">Experiment 3: Expert Configuration Search</h3>
                <p className="text-slate-300 text-sm mb-3">
                  Optimize expert count, learning rates, and top-k values for DeepSeek attention + GLM4 MoE hybrid models.
                </p>
                <span className="inline-block bg-slate-700/50 text-slate-300 text-xs px-2 py-1 rounded">Validation Metrics</span>
              </div>
            </div>
          </section>

          {/* Getting Started */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-6 text-white">Getting Started</h2>
            <div className="bg-slate-800/30 border border-slate-600/30 rounded-lg p-6">
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2 text-green-400">1. Clone the Repository</h3>
                  <div className="bg-slate-900/50 rounded-lg p-3 font-mono text-sm text-slate-300">
                    git clone https://github.com/vukrosic/zero-to-ai-researcher
                  </div>
                </div>
                
                <div>
                  <h3 className="font-semibold mb-2 text-blue-400">2. Install Dependencies</h3>
                  <div className="bg-slate-900/50 rounded-lg p-3 font-mono text-sm text-slate-300">
                    pip install -r requirements.txt
                  </div>
                </div>
                
                <div>
                  <h3 className="font-semibold mb-2 text-purple-400">3. Start Learning</h3>
                  <div className="bg-slate-900/50 rounded-lg p-3 font-mono text-sm text-slate-300">
                    cd _course/01_python_beginner_lessons
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Call to Action */}
          <section className="text-center">
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
        </article>
      </main>
    </div>
  );
}
