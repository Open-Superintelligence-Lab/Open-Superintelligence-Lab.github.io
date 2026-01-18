'use client';

import Link from "next/link";
import VaporwaveGrid from "@/components/vaporwave-grid";

export default function Home() {
  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden min-h-screen flex flex-col">
        {/* Vaporwave Grid Animation */}
        <VaporwaveGrid />

        {/* Floating geometric shapes - Removed for cleaner look */}
        {/* <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-20 left-10 w-20 h-20 border border-blue-500/20 rounded-2xl rotate-12 animate-float"></div>
          <div className="absolute top-40 right-20 w-16 h-16 border border-purple-500/20 rounded-full animate-float-delayed"></div>
          <div className="absolute bottom-40 left-1/4 w-24 h-24 border border-cyan-500/20 rotate-45 animate-float"></div>
          <div className="absolute bottom-20 right-1/3 w-14 h-14 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-lg rotate-6 animate-float-delayed"></div>
        </div> */}

        <div className="relative z-10 container mx-auto px-6 mt-[49vh] pb-20">
          <div className="text-center max-w-6xl mx-auto">
            {/* Main Heading with modern gradient */}
            <div className="relative mb-4 pb-3 overflow-visible">
              <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.4] pb-4"
                  style={{
                    filter: 'drop-shadow(0 2px 8px rgba(0, 0, 0, 0.8)) drop-shadow(0 0 2px rgba(0, 0, 0, 0.9)) drop-shadow(0 -1px 0 rgba(255, 255, 255, 0.1))'
                  }}>
                <span className="inline-block bg-gradient-to-b from-foreground via-foreground to-muted-foreground bg-clip-text text-transparent" 
                      style={{ 
                        paddingBottom: '0.52rem'
                      }}>
                  Open Superintelligence Lab
                </span>
              </h1>
            </div>

            {/* Subtitle with embossed text shadow for readability */}
            <p className="text-lg md:text-xl lg:text-2xl text-muted-foreground mb-16 md:mb-8 font-medium"
               style={{
                 textShadow: '0 2px 8px rgba(0, 0, 0, 0.8), 0 0 2px rgba(0, 0, 0, 0.9), 0 1px 0 rgba(255, 255, 255, 0.1)'
               }}>
              Do AI research with top tier AI researchers
            </p>

            {/* CTA Buttons - properly centered with gap between them */}
            <div className="flex flex-col sm:flex-row justify-center items-center mb-16" style={{ gap: '1rem', transform: 'translateX(-12.5px)' }}>
              <button
                onClick={() => document.getElementById('research-projects')?.scrollIntoView({ behavior: 'smooth' })}
                className="group relative px-8 py-4 bg-gradient-to-r from-gradient-accent-1 via-gradient-accent-2 to-gradient-accent-3 text-primary-foreground font-semibold rounded-xl overflow-hidden transition-all duration-300 hover:shadow-2xl hover:shadow-gradient-accent-2/25"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-gradient-accent-3 via-gradient-accent-1 to-gradient-accent-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <span className="relative flex items-center gap-2">
                  Explore Research
                  <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </span>
              </button>

              <Link
                href="/about"
                className="group px-8 py-4 bg-card/50 backdrop-blur-sm border-2 border-border/50 text-muted-foreground font-semibold rounded-xl hover:border-gradient-accent-1/50 hover:bg-card/80 hover:text-foreground transition-all duration-300"
              >
                <span className="flex items-center gap-2">
                  Learn More
                  <svg className="w-5 h-5 group-hover:rotate-45 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </span>
              </Link>
            </div>

            {/* Scroll indicator - below buttons */}
            <div className="flex justify-center animate-bounce mt-[20vh] md:mt-[24vh]">
              <svg className="w-6 h-6 text-foreground/60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </div>
          </div>
        </div>
      </section>

      {/* Main Projects Section */}
      <main id="research-projects" className="relative bg-gradient-to-b from-gradient-start via-gradient-mid to-gradient-end py-24">
        {/* Background effects - theme aware */}
        <div className="absolute inset-0 bg-[linear-gradient(hsl(var(--foreground)/0.01)_1px,transparent_1px),linear-gradient(90deg,hsl(var(--foreground)/0.01)_1px,transparent_1px)] bg-[size:64px_64px]"></div>

        <div className="relative container mx-auto px-6">
          {/* Section Header */}
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-gradient-accent-1 via-gradient-accent-2 to-gradient-accent-3 bg-clip-text text-transparent">
              Research Projects
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Explore our cutting-edge AI research initiatives pushing the boundaries of what&apos;s possible
            </p>
          </div>

          {/* Projects Grid - Bento-style layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-7xl mx-auto">
            {/* Path to Open Superintelligence - Featured */}
            <Link
              href="/blog/path-to-open-superintelligence"
              className="group relative lg:col-span-2 bg-gradient-to-br from-card/40 to-background/40 backdrop-blur-xl border border-border/50 rounded-2xl p-8 hover:border-accent/50 transition-all duration-500 overflow-hidden"
            >
              {/* Animated gradient border glow */}
              <div className="absolute inset-0 bg-gradient-to-r from-amber-500/0 via-orange-500/0 to-rose-500/0 group-hover:from-amber-500/10 group-hover:via-orange-500/10 group-hover:to-rose-500/10 rounded-2xl transition-all duration-500"></div>

              {/* Glow effect on hover */}
              <div className="absolute -inset-px bg-gradient-to-r from-amber-500 via-orange-500 to-rose-500 rounded-2xl opacity-0 group-hover:opacity-20 blur-xl transition-opacity duration-500"></div>

              <div className="relative">
                <div className="flex flex-wrap items-center gap-2 mb-4">
                  <span className="px-3 py-1 bg-amber-500/20 text-amber-400 text-xs font-semibold rounded-full border border-amber-500/30">
                    Strategic Vision
                  </span>
                  <span className="px-3 py-1 bg-muted/50 text-muted-foreground text-xs font-medium rounded-full">
                    Featured
                  </span>
                </div>

                <h3 className="text-3xl font-bold mb-4 group-hover:text-amber-400 transition-colors">
                  Path To Open Superintelligence
                </h3>

                <p className="text-muted-foreground text-base mb-6 leading-relaxed">
                  A strategic roadmap for building AGI through open collaboration, addressing key challenges and defining our path forward to create transformative AI systems that benefit all of humanity.
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <span>Vision Document</span>
                  </div>
                  <span className="flex items-center gap-2 text-amber-400 text-sm font-medium group-hover:gap-3 transition-all">
                    Read More
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </span>
                </div>
              </div>
            </Link>

            {/* DeepSeek Sparse Attention Project */}
            <Link
              href="/blog/deepseek-sparse-attention"
              className="group relative bg-gradient-to-br from-card/40 to-background/40 backdrop-blur-xl border border-border/50 rounded-2xl p-7 hover:border-gradient-accent-1/50 transition-all duration-500 overflow-hidden"
            >
              <div className="absolute -inset-px bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl opacity-0 group-hover:opacity-20 blur-xl transition-opacity duration-500"></div>

              <div className="relative">
                <div className="flex flex-wrap items-center gap-2 mb-4">
                  <span className="px-3 py-1 bg-blue-500/20 text-blue-400 text-xs font-semibold rounded-full border border-blue-500/30">
                    DeepSeek Research
                  </span>
                </div>

                <h4 className="text-2xl font-bold mb-3 group-hover:text-blue-400 transition-colors">
                  DeepSeek Sparse Attention - DeepSeek-V3.2-Exp
                </h4>

                <p className="text-muted-foreground text-sm mb-5 leading-relaxed">
                  Advanced research on DeepSeek&apos;s innovative sparse attention mechanisms for efficient long-context processing and memory optimization
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    <span>Research Paper</span>
                  </div>
                  <span className="text-blue-400 text-sm font-medium group-hover:translate-x-1 transition-transform inline-block">
                    →
                  </span>
                </div>
              </div>
            </Link>

            {/* Tiny Recursive Model Project */}
            <Link
              href="/blog/tiny-recursive-model"
              className="group relative bg-gradient-to-br from-card/40 to-background/40 backdrop-blur-xl border border-border/50 rounded-2xl p-7 hover:border-gradient-accent-2/50 transition-all duration-500 overflow-hidden"
            >
              <div className="absolute -inset-px bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl opacity-0 group-hover:opacity-20 blur-xl transition-opacity duration-500"></div>

              <div className="relative">
                <div className="flex flex-wrap items-center gap-2 mb-4">
                  <span className="px-3 py-1 bg-purple-500/20 text-purple-400 text-xs font-semibold rounded-full border border-purple-500/30">
                    Latest Research
                  </span>
                </div>

                <h4 className="text-2xl font-bold mb-3 group-hover:text-purple-400 transition-colors">
                  Tiny Recursive Model
                </h4>

                <p className="text-slate-400 text-sm mb-5 leading-relaxed">
                  How a 7M parameter model beats 100x bigger models at Sudoku, Mazes, and ARC-AGI using recursive reasoning with a 2-layer transformer
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                    </svg>
                    <span>AI Architecture</span>
                  </div>
                  <span className="text-purple-400 text-sm font-medium group-hover:translate-x-1 transition-transform inline-block">
                    →
                  </span>
                </div>
              </div>
            </Link>

            {/* Pretrain LLM with NVFP4 Project */}
            <Link
              href="/blog/pretrain-llm-with-nvfp4"
              className="group relative bg-gradient-to-br from-card/40 to-background/40 backdrop-blur-xl border border-border/50 rounded-2xl p-7 hover:border-accent/50 transition-all duration-500 overflow-hidden"
            >
              <div className="absolute -inset-px bg-gradient-to-r from-green-500 to-emerald-500 rounded-2xl opacity-0 group-hover:opacity-20 blur-xl transition-opacity duration-500"></div>

              <div className="relative">
                <div className="flex flex-wrap items-center gap-2 mb-4">
                  <span className="px-3 py-1 bg-green-500/20 text-green-400 text-xs font-semibold rounded-full border border-green-500/30">
                    NVIDIA Research
                  </span>
                </div>

                <h4 className="text-2xl font-bold mb-3 group-hover:text-green-400 transition-colors">
                  Pretrain LLM with NVFP4
                </h4>

                <p className="text-slate-400 text-sm mb-5 leading-relaxed">
                  NVIDIA&apos;s breakthrough 4-bit training methodology achieving 2-3x speedup and 50% memory reduction without sacrificing model quality
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    <span>Training Innovation</span>
                  </div>
                  <span className="text-green-400 text-sm font-medium group-hover:translate-x-1 transition-transform inline-block">
                    →
                  </span>
                </div>
              </div>
            </Link>

            {/* Diffusion Transformer RAE Project */}
            <Link
              href="/blog/diffusion-transformer-representation-autoencoder"
              className="group relative bg-gradient-to-br from-card/40 to-background/40 backdrop-blur-xl border border-border/50 rounded-2xl p-7 hover:border-gradient-accent-3/50 transition-all duration-500 overflow-hidden"
            >
              <div className="absolute -inset-px bg-gradient-to-r from-cyan-500 to-blue-500 rounded-2xl opacity-0 group-hover:opacity-20 blur-xl transition-opacity duration-500"></div>

              <div className="relative">
                <div className="flex flex-wrap items-center gap-2 mb-4">
                  <span className="px-3 py-1 bg-cyan-500/20 text-cyan-400 text-xs font-semibold rounded-full border border-cyan-500/30">
                    MIT-Han Lab
                  </span>
                </div>

                <h4 className="text-2xl font-bold mb-3 group-hover:text-cyan-400 transition-colors">
                  47x Faster Image Generation Training
                </h4>

                <p className="text-slate-400 text-sm mb-5 leading-relaxed">
                  Diffusion Transformers with Representation Autoencoders achieve state-of-the-art FID 1.13 on ImageNet while training 47x faster (80 vs 1400 epochs)
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span>Image Generation</span>
                  </div>
                  <span className="text-cyan-400 text-sm font-medium group-hover:translate-x-1 transition-transform inline-block">
                    →
                  </span>
                </div>
              </div>
            </Link>

            {/* QeRL Project */}
            <Link
              href="/blog/qerl-quantization-reinforcement-learning"
              className="group relative bg-gradient-to-br from-card/40 to-background/40 backdrop-blur-xl border border-border/50 rounded-2xl p-7 hover:border-accent/50 transition-all duration-500 overflow-hidden"
            >
              <div className="absolute -inset-px bg-gradient-to-r from-orange-500 to-rose-500 rounded-2xl opacity-0 group-hover:opacity-20 blur-xl transition-opacity duration-500"></div>

              <div className="relative">
                <div className="flex flex-wrap items-center gap-2 mb-4">
                  <span className="px-3 py-1 bg-orange-500/20 text-orange-400 text-xs font-semibold rounded-full border border-orange-500/30">
                    MIT-Han Lab
                  </span>
                </div>

                <h4 className="text-2xl font-bold mb-3 group-hover:text-orange-400 transition-colors">
                  QeRL: Beyond Efficiency
                </h4>

                <p className="text-slate-400 text-sm mb-5 leading-relaxed">
                  Quantization-enhanced Reinforcement Learning for LLMs achieves 1.5x speedup and enables RL training of 32B models on a single H100 80GB GPU
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    <span>Reinforcement Learning</span>
                  </div>
                  <span className="text-orange-400 text-sm font-medium group-hover:translate-x-1 transition-transform inline-block">
                    →
                  </span>
                </div>
              </div>
            </Link>
          </div>
        </div>
      </main>
    </>
  );
}
