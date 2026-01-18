'use client';

import Link from "next/link";

export default function Home() {
  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden min-h-screen flex flex-col pt-20">
        {/* Animated mesh gradient background */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950"></div>

        {/* Animated gradient mesh overlay */}
        <div className="absolute inset-0 opacity-40">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/30 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-purple-500/30 rounded-full blur-3xl animate-pulse delay-700"></div>
          <div className="absolute bottom-1/4 left-1/3 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        </div>

        {/* Grid pattern overlay */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.02)_1px,transparent_1px)] bg-[size:72px_72px]"></div>

        {/* Floating geometric shapes - Removed for cleaner look */}
        {/* <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-20 left-10 w-20 h-20 border border-blue-500/20 rounded-2xl rotate-12 animate-float"></div>
          <div className="absolute top-40 right-20 w-16 h-16 border border-purple-500/20 rounded-full animate-float-delayed"></div>
          <div className="absolute bottom-40 left-1/4 w-24 h-24 border border-cyan-500/20 rotate-45 animate-float"></div>
          <div className="absolute bottom-20 right-1/3 w-14 h-14 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-lg rotate-6 animate-float-delayed"></div>
        </div> */}

        <div className="relative container mx-auto px-6 py-20 flex-grow flex items-center">
          <div className="text-center max-w-6xl mx-auto">
            {/* Main Heading with modern gradient */}
            <div className="relative mb-8">
              <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold mb-6 tracking-tight">
                <span className="inline-block bg-gradient-to-b from-white via-white to-slate-400 bg-clip-text text-transparent">
                  Open Superintelligence Lab
                </span>
              </h1>
            </div>

            {/* Subtitle with typing effect styling */}
            <p className="text-lg md:text-xl lg:text-2xl text-slate-300 mb-8 font-medium">
              Do AI research with top tier AI researchers
            </p>

            {/* Description */}
            {/* <p className="text-base md:text-lg text-slate-400 mb-12 max-w-2xl mx-auto leading-relaxed">
              Building AGI through open collaboration. Join researchers, engineers, and visionaries advancing AI research and development.
            </p> */}

            {/* Feature Tags */}
            {/* <div className="flex flex-wrap justify-center gap-3 mb-12">
              {[
                { icon: '🌐', text: 'Open Source', color: 'blue' },
                { icon: '🧠', text: 'LLM Research', color: 'purple' },
                { icon: '⚡', text: 'Innovation', color: 'cyan' },
                { icon: '🤝', text: 'Collaboration', color: 'pink' },
              ].map((tag, i) => (
                <div
                  key={i}
                  className={`group relative px-4 py-2 bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 rounded-full hover:border-${tag.color}-500/50 transition-all duration-300 hover:scale-105`}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{tag.icon}</span>
                    <span className="text-sm font-medium text-slate-300 group-hover:text-white">
                      {tag.text}
                    </span>
                  </div>
                </div>
              ))}
            </div> */}

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <button
                onClick={() => document.getElementById('research-projects')?.scrollIntoView({ behavior: 'smooth' })}
                className="px-8 py-4 bg-white text-slate-950 font-semibold rounded-full hover:bg-slate-200 transition-colors duration-300"
              >
                <span className="flex items-center gap-2">
                  Explore Research
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                  </svg>
                </span>
              </button>

              <Link
                href="/about"
                className="px-8 py-4 text-slate-300 font-semibold rounded-full hover:text-white transition-colors duration-300"
              >
                Learn More
              </Link>
            </div>

            {/* Scroll indicator */}
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce opacity-50">
              <svg className="w-6 h-6 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </div>
          </div>
        </div>
      </section>

      {/* Main Projects Section */}
      <main id="research-projects" className="relative bg-slate-950 py-24">

        <div className="relative container mx-auto px-6">
          {/* Section Header */}
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">
              Research Projects
            </h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Explore our cutting-edge AI research initiatives.
            </p>
          </div>

          {/* Projects Grid - Cleaner layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-7xl mx-auto">
            {/* Path to Open Superintelligence - Featured */}
            <Link
              href="/blog/path-to-open-superintelligence"
              className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-xl p-8 hover:bg-slate-900 hover:border-slate-700 transition-all duration-300"
            >
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <span className="text-amber-500 text-xs font-semibold uppercase tracking-wider">
                  Strategic Vision
                </span>
              </div>

              <h3 className="text-2xl font-semibold mb-3 text-slate-100">
                Path To Open Superintelligence
              </h3>

              <p className="text-slate-400 text-base mb-6 leading-relaxed max-w-3xl">
                A strategic roadmap for building AGI through open collaboration, addressing key challenges and defining our path forward to create transformative AI systems.
              </p>

              <div className="flex items-center text-sm text-slate-500">
                <span>Read Vision Document &rarr;</span>
              </div>
            </Link>

            {/* DeepSeek Sparse Attention Project */}
            <Link
              href="/blog/deepseek-sparse-attention"
              className="bg-slate-900/50 border border-slate-800 rounded-xl p-8 hover:bg-slate-900 hover:border-slate-700 transition-all duration-300"
            >
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <span className="text-blue-500 text-xs font-semibold uppercase tracking-wider">
                  DeepSeek Research
                </span>
              </div>

              <h4 className="text-xl font-semibold mb-3 text-slate-100">
                DeepSeek Sparse Attention
              </h4>

              <p className="text-slate-400 text-sm mb-6 leading-relaxed">
                Advanced research on DeepSeek&apos;s innovative sparse attention mechanisms for efficient long-context processing.
              </p>

              <div className="flex items-center text-sm text-slate-500">
                <span>Read Paper &rarr;</span>
              </div>
            </Link>

            {/* Tiny Recursive Model Project */}
            <Link
              href="/blog/tiny-recursive-model"
              className="bg-slate-900/50 border border-slate-800 rounded-xl p-8 hover:bg-slate-900 hover:border-slate-700 transition-all duration-300"
            >
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <span className="text-purple-500 text-xs font-semibold uppercase tracking-wider">
                  Latest Research
                </span>
              </div>

              <h4 className="text-xl font-semibold mb-3 text-slate-100">
                Tiny Recursive Model
              </h4>

              <p className="text-slate-400 text-sm mb-6 leading-relaxed">
                How a 7M parameter model beats 100x bigger models at Sudoku, Mazes, and ARC-AGI using recursive reasoning.
              </p>

              <div className="flex items-center text-sm text-slate-500">
                <span>Read Architecture &rarr;</span>
              </div>
            </Link>

            {/* Pretrain LLM with NVFP4 Project */}
            <Link
              href="/blog/pretrain-llm-with-nvfp4"
              className="bg-slate-900/50 border border-slate-800 rounded-xl p-8 hover:bg-slate-900 hover:border-slate-700 transition-all duration-300"
            >
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <span className="text-green-500 text-xs font-semibold uppercase tracking-wider">
                  NVIDIA Research
                </span>
              </div>

              <h4 className="text-xl font-semibold mb-3 text-slate-100">
                Pretrain LLM with NVFP4
              </h4>

              <p className="text-slate-400 text-sm mb-6 leading-relaxed">
                NVIDIA&apos;s breakthrough 4-bit training methodology achieving 2-3x speedup and 50% memory reduction.
              </p>

              <div className="flex items-center text-sm text-slate-500">
                <span>Read Innovation &rarr;</span>
              </div>
            </Link>

            {/* Diffusion Transformer RAE Project */}
            <Link
              href="/blog/diffusion-transformer-representation-autoencoder"
              className="bg-slate-900/50 border border-slate-800 rounded-xl p-8 hover:bg-slate-900 hover:border-slate-700 transition-all duration-300"
            >
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <span className="text-cyan-500 text-xs font-semibold uppercase tracking-wider">
                  MIT-Han Lab
                </span>
              </div>

              <h4 className="text-xl font-semibold mb-3 text-slate-100">
                47x Faster Image Generation
              </h4>

              <p className="text-slate-400 text-sm mb-6 leading-relaxed">
                Diffusion Transformers with Representation Autoencoders achieve state-of-the-art FID 1.13 on ImageNet.
              </p>

              <div className="flex items-center text-sm text-slate-500">
                <span>Read Research &rarr;</span>
              </div>
            </Link>

            {/* QeRL Project */}
            <Link
              href="/blog/qerl-quantization-reinforcement-learning"
              className="bg-slate-900/50 border border-slate-800 rounded-xl p-8 hover:bg-slate-900 hover:border-slate-700 transition-all duration-300"
            >
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <span className="text-orange-500 text-xs font-semibold uppercase tracking-wider">
                  MIT-Han Lab
                </span>
              </div>

              <h4 className="text-xl font-semibold mb-3 text-slate-100">
                QeRL: Beyond Efficiency
              </h4>

              <p className="text-slate-400 text-sm mb-6 leading-relaxed">
                Quantization-enhanced Reinforcement Learning for LLMs enables RL training of 32B models on a single GPU.
              </p>

              <div className="flex items-center text-sm text-slate-500">
                <span>Read Research &rarr;</span>
              </div>
            </Link>
          </div>
        </div>
      </main>
    </>
  );
}
