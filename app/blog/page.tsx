'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
    BookOpen,
    Code2,
    ArrowUpRight,
    Sparkles,
    Layers,
    Zap,
    CheckCircle2,
    Users,
    Clock,
    Award,
    Youtube,
    UserCheck,
    Globe
} from 'lucide-react';

const ImplementationChallenge = ({
    title,
    description,
    papers,
    difficulty,
    tags
}: {
    title: string,
    description: string,
    papers: { name: string, url?: string }[],
    difficulty: 'Mainstream' | 'Advanced' | 'Expert',
    tags: string[]
}) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="group relative p-8 rounded-3xl bg-slate-900/40 border border-white/5 hover:border-blue-500/30 transition-all duration-300 backdrop-blur-sm overflow-hidden"
    >
        <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
            <Code2 className="w-24 h-24" />
        </div>

        <div className="relative">
            <div className="flex justify-between items-start mb-6">
                <div className="flex gap-2">
                    {tags.map(tag => (
                        <span key={tag} className="px-2 py-1 rounded-md bg-white/5 border border-white/10 text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                            {tag}
                        </span>
                    ))}
                </div>
                <span className={`text-[10px] font-bold px-2 py-1 rounded uppercase tracking-widest border ${difficulty === 'Expert' ? 'border-red-500/30 text-red-400 bg-red-500/5' :
                    difficulty === 'Advanced' ? 'border-amber-500/30 text-amber-400 bg-amber-500/5' :
                        'border-blue-500/30 text-blue-400 bg-blue-500/5'
                    }`}>
                    {difficulty}
                </span>
            </div>

            <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-blue-400 transition-colors tracking-tight">
                {title}
            </h3>

            <p className="text-slate-400 text-sm leading-relaxed mb-8 max-w-xl">
                {description}
            </p>

            <div className="space-y-4 mb-8">
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
                    <BookOpen className="w-3 h-3" />
                    Reference Papers
                </h4>
                <div className="grid gap-2">
                    {papers.map((paper, i) => (
                        paper.url ? (
                            <a
                                key={i}
                                href={paper.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 group/paper hover:bg-white/10 transition-colors cursor-pointer"
                            >
                                <span className="text-sm text-slate-300 font-light">{paper.name}</span>
                                <ArrowUpRight className="w-4 h-4 text-slate-500 group-hover/paper:text-blue-400 transition-colors" />
                            </a>
                        ) : (
                            <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 group/paper transition-colors">
                                <span className="text-sm text-slate-300 font-light">{paper.name}</span>
                                <ArrowUpRight className="w-4 h-4 text-slate-500/20" />
                            </div>
                        )
                    ))}
                </div>
            </div>

        </div>
    </motion.div>
);

export default function PublishPage() {
    return (
        <div className="min-h-screen bg-[#050505] text-slate-200">
            {/* Background decoration */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-blue-600/5 rounded-full blur-[120px]" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-purple-600/5 rounded-full blur-[120px]" />
            </div>

            <div className="relative container mx-auto px-6 pt-40 pb-24">
                {/* Header */}
                <div className="max-w-4xl mb-24">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] font-bold uppercase tracking-[0.2em] mb-8">
                            <Sparkles className="w-3 h-3" />
                            Open Research Call
                        </div>
                        <h1 className="text-6xl md:text-8xl font-bold text-white mb-8 tracking-tighter leading-[0.9]">
                            Publish <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400">Research.</span>
                        </h1>
                        <p className="text-xl text-slate-400 leading-relaxed max-w-2xl font-light">
                            Write a blog post and get <span className="text-white">free mentorship, experience, credit, and publication</span>.
                            Get direct feedback, a <span className="text-white">YouTube mention</span>, and contribute to science.
                        </p>
                    </motion.div>
                </div>

                {/* Benefits Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-24">
                    {[
                        { icon: Users, title: "Free Mentorship", desc: "Direct guidance and feedback on your research and writing." },
                        { icon: Award, title: "Scientific Credit", desc: "Get full credit for your implementation and insights." },
                        { icon: Globe, title: "Contribution", desc: "Add to the collective understanding of AI." },
                        { icon: Youtube, title: "YouTube Mention", desc: "Featured in our video." },
                        { icon: Clock, title: "Rapid Iteration", desc: "Fast feedback loops to refine your work." },
                        { icon: UserCheck, title: "Personal Growth", desc: "Level up your experience." }
                    ].map((item, i) => (
                        <div key={i} className="p-6 rounded-2xl bg-white/[0.02] border border-white/5 hover:border-blue-500/20 transition-all">
                            <item.icon className="w-6 h-6 text-blue-400 mb-4" />
                            <h4 className="text-lg font-bold text-white mb-2">{item.title}</h4>
                            <p className="text-sm text-slate-500 leading-relaxed">{item.desc}</p>
                        </div>
                    ))}
                </div>

                {/* The Rules Section */}
                <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    className="mb-24 p-10 rounded-3xl bg-gradient-to-br from-blue-600/5 to-purple-600/5 border border-white/5 backdrop-blur-sm"
                >
                    <h2 className="text-3xl font-bold text-white mb-8">The Process</h2>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                        <div className="space-y-6">
                            <p className="text-slate-300 leading-relaxed">
                                You would be responsible for most of the work, with my direct help and feedback.
                                The goal is to produce high-signal content that advances the field.
                            </p>
                            <div className="p-6 rounded-2xl bg-amber-500/5 border border-amber-500/10">
                                <div className="flex items-center gap-3 mb-3 text-amber-400">
                                    <CheckCircle2 className="w-5 h-5" />
                                    <h4 className="font-bold">The Golden Rule</h4>
                                </div>
                                <p className="text-sm text-slate-400">
                                    <span className="text-white font-medium">No AI generated text.</span> AI may be used as a tool during creation,
                                    but the final blog must be human-written.
                                </p>
                            </div>
                        </div>
                        <div className="space-y-8">
                            <div>
                                <h4 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                                    <Zap className="w-4 h-4 text-blue-400" />
                                    AI Research Post
                                </h4>
                                <ul className="space-y-2 text-sm text-slate-400">
                                    <li className="flex items-center gap-2">• Select a relevant research topic</li>
                                    <li className="flex items-center gap-2">• Code small, high-signal experiments</li>
                                    <li className="flex items-center gap-2">• Describe math background (optional), experiments, and results</li>
                                </ul>
                            </div>
                            <div>
                                <h4 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                                    <BookOpen className="w-4 h-4 text-purple-400" />
                                    Tutorial / Explanation
                                </h4>
                                <ul className="space-y-2 text-sm text-slate-400">
                                    <li className="flex items-center gap-2">• Write a detailed explanation of a research paper</li>
                                    <li className="flex items-center gap-2">• Detail specific experiments or mathematical concepts</li>
                                    <li className="flex items-center gap-2">• Create guides that we wish existed</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    <div className="mt-12 pt-8 border-t border-white/5">
                        <p className="text-center text-slate-500 italic text-sm">
                            &ldquo;Author has total freedom to choose topics.&rdquo;
                        </p>
                    </div>
                </motion.div>

                {/* Modular Tracks Section */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="mb-20 p-8 rounded-3xl bg-white/[0.01] border border-white/5 transition-all"
                >
                    <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                        <Layers className="w-6 h-6 text-blue-400" />
                        Possible Contributions
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {[
                            { title: "Explain the Paper", desc: "Translate high-level concepts and intuition into clear, narrative explanations." },
                            { title: "Background Math", desc: "Deep-dive into the formal mathematical foundations and proofs." },
                            { title: "Implement Code", desc: "Write clean, modular PyTorch or Triton code for the research logic." },
                            { title: "Training Dataset", desc: "Build the data pipelines and synthetic environments needed for experiments." },
                            { title: "Empirical Analysis", desc: "Run experiments, collect insights, and visualize the results." },
                            { title: "Peer Review", desc: "Collaborate on existing work and provide technical feedback." }
                        ].map((track, i) => (
                            <div key={i} className="flex gap-4 items-start p-4 rounded-xl border border-white/5">
                                <div className="mt-1 w-5 h-5 rounded-full bg-blue-500/10 border border-blue-500/30 flex items-center justify-center flex-shrink-0">
                                    <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                                </div>
                                <div>
                                    <h4 className="text-sm font-bold text-white mb-1">{track.title}</h4>
                                    <p className="text-xs text-slate-500 leading-relaxed font-light">{track.desc}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </motion.div>

                {/* Challenge Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-24">
                    <ImplementationChallenge
                        title="LeJEPA Scaling & Stability"
                        description="Implementing the Latent-Euclidean Joint-Embedding Predictive Architecture. Focus on SIGReg loss and proving stability."
                        difficulty="Advanced"
                        tags={['SSL', 'JEP-A', 'PyTorch']}
                        papers={[{ name: "LeJEPA: Provable and Scalable SSL", url: "https://arxiv.org/pdf/2511.08544" }]}
                    />
                    <ImplementationChallenge
                        title="Continual Learning at Scale"
                        description="Implementation of Google's research on preventing catastrophic forgetting in transformers."
                        difficulty="Expert"
                        tags={['Optimization', 'Transformers']}
                        papers={[{ name: "Continual Learning by Google Research" }]}
                    />
                </div>
            </div>
        </div>
    );
}
