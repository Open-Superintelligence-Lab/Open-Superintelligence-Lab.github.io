'use client';

import Link from "next/link";

export default function DeepDeltaLearningProject() {
    return (
        <>
            {/* Hero Section */}
            <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
                {/* Background effects */}
                <div className="absolute inset-0 bg-gradient-to-r from-violet-600/20 via-purple-600/20 to-fuchsia-600/20"></div>
                <div className="absolute inset-0 opacity-30">
                    <div className="absolute inset-0 bg-gradient-to-br from-transparent via-purple-500/5 to-transparent"></div>
                </div>

                {/* Animated background particles */}
                <div className="absolute inset-0 overflow-hidden">
                    <div className="absolute top-1/6 left-1/6 w-3 h-3 bg-gradient-to-r from-violet-400 to-purple-400 rounded-full opacity-60 animate-pulse"></div>
                    <div className="absolute top-1/4 right-1/5 w-2 h-2 bg-gradient-to-r from-purple-400 to-fuchsia-400 rounded-full opacity-50 animate-pulse delay-300"></div>
                    <div className="absolute top-1/3 left-1/8 w-4 h-4 bg-gradient-to-r from-fuchsia-400 to-violet-400 rounded-full opacity-40 animate-pulse delay-700"></div>
                    <div className="absolute bottom-1/4 right-1/6 w-2.5 h-2.5 bg-gradient-to-r from-violet-400 to-purple-400 rounded-full opacity-55 animate-pulse delay-1000"></div>
                </div>

                <div className="relative container mx-auto px-6 pt-32 pb-12">
                    <div className="text-center max-w-4xl mx-auto">
                        <div className="relative">
                            <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium mb-8 leading-tight">
                                <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 bg-clip-text text-transparent">
                                    Deep Delta Learning
                                </span>
                            </h1>
                            <div className="text-lg md:text-xl text-slate-400 mb-8">
                                🔬 Novel Approach to Neural Architecture Residual Updates
                            </div>

                            {/* Tags */}
                            <div className="flex items-center justify-center gap-3 text-sm text-slate-400 mb-8">
                                <span className="flex items-center gap-2">
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                    </svg>
                                    Research Implementation
                                </span>
                                <span className="text-slate-600">•</span>
                                <span className="flex items-center gap-2">
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                    Research Article
                                </span>
                            </div>

                            {/* Glow effect for the title */}
                            <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-medium leading-tight blur-sm">
                                <span className="bg-gradient-to-r from-violet-400/20 via-purple-400/20 to-fuchsia-400/20 bg-clip-text text-transparent">
                                    Deep Delta Learning
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
                            {/* Article Body */}
                            <div className="px-8 sm:px-12 py-20">
                                <div className="prose prose-lg prose-invert max-w-none">
                                    {/* Overview */}
                                    <div className="mb-12">
                                        <h2 className="text-3xl font-bold text-white mb-6">Overview</h2>
                                        <p className="text-slate-300 leading-relaxed mb-4">
                                            Deep Delta Learning (DDL) is a novel approach to neural architecture that replaces traditional
                                            additive residuals with a learned update mechanism. This research repository is dedicated to
                                            exploring and optimizing this breakthrough architecture.
                                        </p>
                                        <p className="text-slate-300 leading-relaxed">
                                            The core innovation is the <strong className="text-purple-400">Deep Delta Residual (DeepDeltaRes)</strong>,
                                            which learns how to update representations rather than simply adding them.
                                        </p>
                                    </div>

                                    {/* Research Paper */}
                                    <div className="mb-12 p-6 bg-purple-500/10 border border-purple-500/20 rounded-xl">
                                        <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                                            <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                            </svg>
                                            Research Paper
                                        </h3>
                                        <a
                                            href="https://arxiv.org/pdf/2601.00417"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-purple-400 hover:text-purple-300 transition-colors flex items-center gap-2"
                                        >
                                            Deep Delta Learning on arXiv
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                            </svg>
                                        </a>
                                    </div>

                                    {/* GitHub Repository */}
                                    <div className="mb-12 p-6 bg-violet-500/10 border border-violet-500/20 rounded-xl">
                                        <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                                            <svg className="w-6 h-6 text-violet-400" fill="currentColor" viewBox="0 0 24 24">
                                                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                                            </svg>
                                            Implementation Repository
                                        </h3>
                                        <a
                                            href="https://github.com/vukrosic/deep-delta-learning-research"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-violet-400 hover:text-violet-300 transition-colors flex items-center gap-2 mb-3"
                                        >
                                            vukrosic/deep-delta-learning-research
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                            </svg>
                                        </a>
                                        <p className="text-slate-400 text-sm">
                                            Also available in the <a
                                                href="https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/95/changes"
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="text-violet-400 hover:text-violet-300 transition-colors"
                                            >
                                                5-dollar-llm repository
                                            </a>
                                        </p>
                                    </div>

                                    {/* Research Focus */}
                                    <div className="mb-12">
                                        <h2 className="text-3xl font-bold text-white mb-6">Research Focus</h2>
                                        <p className="text-slate-300 leading-relaxed mb-6">
                                            This repository investigates <strong className="text-purple-400">Deep Delta Residuals (DeepDeltaRes)</strong>,
                                            which replaces traditional additive residuals with a learned update mechanism.
                                        </p>

                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            {[
                                                { title: "Explain the Paper", desc: "Translate high-level concepts and intuition into clear, narrative explanations." },
                                                { title: "Background Math", desc: "Deep-dive into the formal mathematical foundations and proofs." },
                                                { title: "Implement Code", desc: "Write clean, modular PyTorch or Triton code for the research logic." },
                                                { title: "Training Dataset", desc: "Build data pipelines and synthetic environments needed for experiments." },
                                                { title: "Empirical Analysis", desc: "Run experiments, collect insights, and visualize results." },
                                                { title: "Peer Review", desc: "Collaborate on existing work and provide technical feedback." }
                                            ].map((item, i) => (
                                                <div key={i} className="p-4 bg-white/5 border border-white/10 rounded-lg">
                                                    <h4 className="text-lg font-semibold text-purple-400 mb-2">{item.title}</h4>
                                                    <p className="text-slate-400 text-sm">{item.desc}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Getting Started */}
                                    <div className="mb-12">
                                        <h2 className="text-3xl font-bold text-white mb-6">🚀 Getting Started</h2>
                                        <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6 mb-4">
                                            <h4 className="text-lg font-semibold text-white mb-3">1. Environment Setup</h4>
                                            <pre className="bg-slate-900 text-slate-300 p-4 rounded-lg overflow-x-auto">
                                                <code>pip install -r requirements.txt</code>
                                            </pre>
                                        </div>
                                        <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
                                            <h4 className="text-lg font-semibold text-white mb-3">2. Run Research Baseline</h4>
                                            <pre className="bg-slate-900 text-slate-300 p-4 rounded-lg overflow-x-auto mb-3">
                                                <code>python train_llm.py</code>
                                            </pre>
                                            <p className="text-slate-400 text-sm">
                                                This will train a 100M parameter model using Deep Delta Learning on the base dataset.
                                            </p>
                                        </div>
                                    </div>

                                    {/* Repository Structure */}
                                    <div className="mb-12">
                                        <h2 className="text-3xl font-bold text-white mb-6">📂 Repository Structure</h2>
                                        <ul className="space-y-2 text-slate-300">
                                            <li className="flex items-start gap-2">
                                                <span className="text-purple-400 mt-1">•</span>
                                                <span><code className="text-purple-400">models/deepdelta.py</code>: Implementation of the Deep Delta Residual module</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-purple-400 mt-1">•</span>
                                                <span><code className="text-purple-400">models/layers.py</code>: Integration of DDL into Transformer blocks</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-purple-400 mt-1">•</span>
                                                <span><code className="text-purple-400">train_llm.py</code>: Main research training script</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-purple-400 mt-1">•</span>
                                                <span><code className="text-purple-400">configs/</code>: Experiment configurations for model size and hyperparameters</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-purple-400 mt-1">•</span>
                                                <span><code className="text-purple-400">benchmarks/</code>: Evaluation suite for linguistic and logic capabilities</span>
                                            </li>
                                        </ul>
                                    </div>

                                    {/* Research Contributions */}
                                    <div className="mb-12">
                                        <h2 className="text-3xl font-bold text-white mb-6">🤝 Research Contributions</h2>
                                        <p className="text-slate-300 leading-relaxed mb-6">
                                            We welcome theoretical explorations, architectural variants, and optimization strategies for Deep Delta Learning.
                                        </p>
                                        <div className="space-y-4">
                                            <div className="flex items-start gap-3">
                                                <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                                                    <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                    </svg>
                                                </div>
                                                <div>
                                                    <h4 className="text-lg font-semibold text-white mb-1">Focus on Rigor</h4>
                                                    <p className="text-slate-400">All contributions should be backed by experimental results.</p>
                                                </div>
                                            </div>
                                            <div className="flex items-start gap-3">
                                                <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                                                    <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                    </svg>
                                                </div>
                                                <div>
                                                    <h4 className="text-lg font-semibold text-white mb-1">Minimalism</h4>
                                                    <p className="text-slate-400">We prefer clean, modular implementations that isolate the impact of DDL.</p>
                                                </div>
                                            </div>
                                            <div className="flex items-start gap-3">
                                                <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                                                    <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                    </svg>
                                                </div>
                                                <div>
                                                    <h4 className="text-lg font-semibold text-white mb-1">Reproducibility</h4>
                                                    <p className="text-slate-400">Ensure your experiments can be replicated with the provided scripts.</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Article Footer */}
                            <div className="bg-gradient-to-r from-violet-600/5 via-purple-600/5 to-fuchsia-600/5 border-t border-white/10 px-8 sm:px-12 py-8">
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
                                        <a href="https://x.com/intent/tweet?text=Check%20out%20Deep%20Delta%20Learning%20-%20a%20novel%20approach%20to%20neural%20architecture%20residual%20updates%20%F0%9F%94%AC%0A%0AReplacing%20traditional%20additive%20residuals%20with%20learned%20update%20mechanisms.%0A%0A%23AI%20%23MachineLearning%20%23DeepLearning&url=https://opensuperintelligencelab.com/blog/deep-delta-learning/"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-slate-400 hover:text-purple-400 transition-colors">
                                            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                                            </svg>
                                        </a>
                                        <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://opensuperintelligencelab.com/blog/deep-delta-learning/&title=Deep%20Delta%20Learning&summary=A%20novel%20approach%20to%20neural%20architecture%20residual%20updates%20that%20replaces%20traditional%20additive%20residuals%20with%20learned%20update%20mechanisms."
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-slate-400 hover:text-purple-400 transition-colors">
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
                                className="group flex items-center gap-2 px-6 py-3 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-purple-500/50 text-slate-300 hover:text-purple-400 font-medium rounded-xl transition-all duration-300"
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
                                    className="flex items-center gap-1 px-4 py-2 hover:text-purple-400 transition-colors"
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
