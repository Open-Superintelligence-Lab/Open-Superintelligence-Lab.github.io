'use client';

import Link from "next/link";
import { useState } from "react";

// Tooltip Component
function Tooltip({ children, content, position = "top" }: { children: React.ReactNode; content: React.ReactNode; position?: "top" | "bottom" | "left" | "right" }) {
  const [isVisible, setIsVisible] = useState(false);
  const [actualPosition, setActualPosition] = useState(position);

  const positionClasses = {
    top: "bottom-full left-1/2 transform -translate-x-1/2 mb-2",
    bottom: "top-full left-1/2 transform -translate-x-1/2 mt-2",
    left: "right-full top-1/2 transform -translate-y-1/2 mr-2",
    right: "left-full top-1/2 transform -translate-y-1/2 ml-2"
  };

  const handleMouseEnter = () => {
    setIsVisible(true);
    if (position === "top") {
      setActualPosition("right");
    } else {
      setActualPosition(position);
    }
  };

  return (
    <div 
      className="relative inline-block"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <div className={`absolute z-50 ${positionClasses[actualPosition]} pointer-events-none`}>
          <div className="bg-slate-800/95 backdrop-blur-sm border border-slate-600/50 rounded-xl p-4 shadow-2xl max-w-xs w-max">
            <div className="text-white text-sm leading-relaxed">
              {content}
            </div>
            <div className={`absolute w-2 h-2 bg-slate-800/95 border-r border-b border-slate-600/50 transform rotate-45 ${
              actualPosition === "top" ? "top-full left-1/2 -translate-x-1/2 -translate-y-1/2" :
              actualPosition === "bottom" ? "bottom-full left-1/2 -translate-x-1/2 translate-y-1/2" :
              actualPosition === "left" ? "left-full top-1/2 -translate-y-1/2 -translate-x-1/2" :
              "right-full top-1/2 -translate-y-1/2 translate-x-1/2"
            }`}></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function MobileLLMR1Project() {
  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-indigo-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
        </div>
        
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/6 left-1/6 w-3 h-3 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full opacity-60 animate-pulse"></div>
          <div className="absolute top-1/4 right-1/5 w-2 h-2 bg-gradient-to-r from-purple-400 to-indigo-400 rounded-full opacity-50 animate-pulse delay-300"></div>
          <div className="absolute top-1/3 left-1/8 w-4 h-4 bg-gradient-to-r from-indigo-400 to-blue-400 rounded-full opacity-40 animate-pulse delay-700"></div>
          <div className="absolute bottom-1/4 right-1/6 w-2.5 h-2.5 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full opacity-55 animate-pulse delay-1000"></div>
        </div>
        
        <div className="relative container mx-auto px-6 pt-32 pb-24">
          <div className="text-center max-w-4xl mx-auto">
            <div className="relative">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium mb-8 leading-tight">
                <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-indigo-400 bg-clip-text text-transparent">
                  MobileLLM-R1: Small Models, Big Reasoning
                </span>
              </h1>
              <div className="text-lg md:text-xl text-slate-400 mb-4">
                🧠 Sub-billion parameters, trillion-token efficiency
              </div>
              
              <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-medium leading-tight blur-sm">
                <span className="bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-indigo-400/20 bg-clip-text text-transparent">
                  MobileLLM-R1: Small Models, Big Reasoning
                </span>
              </div>
            </div>
            
            <p className="text-xl text-slate-300 mb-12 leading-relaxed">
              How Meta achieved strong reasoning capabilities in sub-billion parameter models using only 2T tokens of high-quality data, challenging the assumption that reasoning requires massive scale
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
        <div className="container mx-auto px-6 max-w-4xl">
          
          {/* TL;DR Section */}
          <div className="mb-8">
            <div className="bg-gradient-to-br from-blue-900/20 to-purple-800/20 backdrop-blur-sm border border-blue-600/30 rounded-xl p-6">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center gap-3">
                <span className="text-4xl">📝</span>
                TL;DR
              </h2>
              <p className="text-slate-300 leading-relaxed mb-4">
                Meta&apos;s MobileLLM-R1 challenges two fundamental assumptions about reasoning in language models: (1) that reasoning only emerges in large models, and (2) that it requires massive datasets. They demonstrate that <strong className="text-blue-400">sub-billion parameter models can achieve strong reasoning</strong> with just 2T tokens of carefully curated data.
              </p>
              <p className="text-slate-300 leading-relaxed">
                Their <strong className="text-purple-400">950M parameter model achieves an AIME score of 15.5</strong>, compared to just 0.6 for OLMo-2-1.48B and 0.3 for SmoILM-2-1.7B. Remarkably, despite being trained on only 11.7% of the tokens compared to Qwen3&apos;s 36T-token corpus, MobileLLM-R1-950M matches or surpasses Qwen3-0.6B across multiple reasoning benchmarks.
              </p>
            </div>
          </div>

          {/* The Challenge */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">⚠️</span>
                The Challenge: Small Models, Big Problems
              </h2>
              <p className="text-slate-400 text-lg">
                Why reasoning in small models is so difficult
              </p>
            </div>
            
            <div className="grid md:grid-cols-3 gap-6 mb-8">
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-orange-400 mb-2">🧠 The Scale Assumption</div>
                    <p className="mb-2">Traditional wisdom suggests reasoning only emerges in models with billions of parameters.</p>
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• GPT-4: ~1.7T parameters</div>
                      <div className="text-xs text-slate-300">• Claude-3: ~1.4T parameters</div>
                      <div className="text-xs text-slate-300">• Reasoning = Large scale?</div>
                    </div>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-orange-900/20 to-orange-800/20 backdrop-blur-sm border border-orange-600/30 rounded-xl p-6 text-center cursor-help hover:border-orange-500/50 transition-all duration-300">
                  <div className="text-4xl mb-4">🧠</div>
                  <h3 className="text-xl font-bold text-white mb-2">Scale Assumption</h3>
                  <div className="text-orange-400 font-mono text-2xl mb-2">Billions</div>
                  <p className="text-slate-300 text-sm">Parameters needed</p>
                </div>
              </Tooltip>
              
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-red-400 mb-2">📊 The Data Assumption</div>
                    <p className="mb-2">Another assumption: reasoning requires training on massive datasets with trillions of tokens.</p>
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• Qwen3: 36T tokens</div>
                      <div className="text-xs text-slate-300">• GPT-4: ~13T tokens</div>
                      <div className="text-xs text-slate-300">• More data = Better reasoning?</div>
                    </div>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-red-900/20 to-red-800/20 backdrop-blur-sm border border-red-600/30 rounded-xl p-6 text-center cursor-help hover:border-red-500/50 transition-all duration-300">
                  <div className="text-4xl mb-4">📊</div>
                  <h3 className="text-xl font-bold text-white mb-2">Data Assumption</h3>
                  <div className="text-red-400 font-mono text-2xl mb-2">Trillions</div>
                  <p className="text-slate-300 text-sm">Tokens required</p>
                </div>
              </Tooltip>
              
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-green-400 mb-2">💡 The Reality</div>
                    <p className="mb-2">MobileLLM-R1 proves both assumptions wrong with smart data curation and training techniques.</p>
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• 950M parameters (sub-billion)</div>
                      <div className="text-xs text-slate-300">• 2T tokens (vs 36T for Qwen3)</div>
                      <div className="text-xs text-slate-300">• Strong reasoning capabilities</div>
                    </div>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-green-900/20 to-emerald-800/20 backdrop-blur-sm border border-green-600/30 rounded-xl p-6 text-center cursor-help hover:border-green-500/50 transition-all duration-300">
                  <div className="text-4xl mb-4">💡</div>
                  <h3 className="text-xl font-bold text-white mb-2">Reality</h3>
                  <div className="text-green-400 font-mono text-2xl mb-2">Smart</div>
                  <p className="text-slate-300 text-sm">Data curation wins</p>
                </div>
              </Tooltip>
            </div>
          </div>

          {/* Key Innovations */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">🔑</span>
                The Key Innovations
              </h2>
              <p className="text-slate-400 text-lg">
                How MobileLLM-R1 achieves reasoning with minimal resources
              </p>
            </div>
            
            <div className="space-y-6">
              {/* Innovation 1 */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-blue-400 mb-2">🎯 Benchmark-Free Data Optimization</div>
                    <p className="mb-2">Instead of optimizing for specific benchmarks, they use self-evolving data optimization that leverages cross-domain influences.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Key Insight</div>
                      <div className="text-xs text-slate-300">Quality over quantity - 2T high-quality tokens beat 36T mixed-quality tokens</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Process</div>
                      <div className="text-xs text-slate-300">1. Identify beneficial datasets using designed metrics</div>
                      <div className="text-xs text-slate-300">2. Curate and resample open-source data</div>
                      <div className="text-xs text-slate-300">3. Create optimal data mixture ratios</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">No benchmark exposure</span> during training</div>
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">Cross-domain learning</span> improves generalization</div>
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">Self-evolving</span> optimization process</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like having a smart tutor who knows exactly what to teach, not just more textbooks!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-blue-900/20 to-purple-800/20 backdrop-blur-sm border border-blue-600/30 rounded-xl p-6 cursor-help hover:border-blue-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold">
                      1
                    </div>
                    <h3 className="text-xl font-bold text-white">Benchmark-Free Data Optimization</h3>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Self-evolving data curation that leverages cross-domain influences to create optimal training mixtures without exposing models to benchmarks during training.
                  </p>
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                    <div className="text-blue-400 text-sm font-mono">2T High-Quality Tokens &gt; 36T Mixed Tokens</div>
                  </div>
                </div>
              </Tooltip>
              
              {/* Innovation 2 */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-purple-400 mb-2">🔄 Data-Model Co-Evolution</div>
                    <p className="mb-2">As the model&apos;s capacity changes during training, the data mixture adapts to match the model&apos;s current capabilities.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Early Training</div>
                      <div className="text-xs text-slate-300">Model capacity: Low → Simple, foundational data</div>
                      <div className="text-xs text-slate-300">Focus: Basic language understanding</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Mid Training</div>
                      <div className="text-xs text-slate-300">Model capacity: Growing → More complex reasoning</div>
                      <div className="text-xs text-slate-300">Focus: Mathematical and logical problems</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Late Training</div>
                      <div className="text-xs text-slate-300">Model capacity: High → Advanced reasoning</div>
                      <div className="text-xs text-slate-300">Focus: Multi-step problem solving</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Adaptive curriculum</span> that grows with the model</div>
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Optimal challenge level</span> at each stage</div>
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Prevents overwhelm</span> or under-stimulation</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like a personal trainer who adjusts the workout as you get stronger!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-purple-900/20 to-indigo-800/20 backdrop-blur-sm border border-purple-600/30 rounded-xl p-6 cursor-help hover:border-purple-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-indigo-600 rounded-lg flex items-center justify-center text-white font-bold">
                      2
                    </div>
                    <h3 className="text-xl font-bold text-white">Data-Model Co-Evolution</h3>
                  </div>
                    <p className="text-slate-300 mb-3">
                    Adaptive training strategy where the data mixture evolves alongside the model&apos;s growing capacity, ensuring optimal challenge levels throughout training.
                  </p>
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                    <div className="text-purple-400 text-sm font-mono">Adaptive Curriculum = Optimal Learning</div>
                  </div>
                </div>
              </Tooltip>
              
              {/* Innovation 3 */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-emerald-400 mb-2">📐 Capability-Aware Data Curation</div>
                    <p className="mb-2">They identify and prioritize data that specifically enhances reasoning capabilities rather than just general language understanding.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Reasoning-Enhancing Data Types</div>
                      <div className="text-xs text-slate-300">• Mathematical problem solving</div>
                      <div className="text-xs text-slate-300">• Logical reasoning chains</div>
                      <div className="text-xs text-slate-300">• Step-by-step explanations</div>
                      <div className="text-xs text-slate-300">• Code generation and debugging</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Quality Metrics</div>
                      <div className="text-xs text-slate-300">• Complexity progression</div>
                      <div className="text-xs text-slate-300">• Reasoning chain length</div>
                      <div className="text-xs text-slate-300">• Cross-domain transferability</div>
                      <div className="text-xs text-slate-300">• Solution correctness</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Targeted selection</span> of reasoning-rich content</div>
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Quality over quantity</span> approach</div>
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Systematic evaluation</span> of data impact</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like choosing the right exercises for building specific muscles!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-emerald-900/20 to-teal-800/20 backdrop-blur-sm border border-emerald-600/30 rounded-xl p-6 cursor-help hover:border-emerald-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-lg flex items-center justify-center text-white font-bold">
                      3
                    </div>
                    <h3 className="text-xl font-bold text-white">Capability-Aware Data Curation</h3>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Systematic identification and prioritization of data that specifically enhances reasoning capabilities, focusing on quality over quantity.
                  </p>
                  <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-3">
                    <div className="text-emerald-400 text-sm font-mono">Reasoning-Specific Data = Better Reasoning</div>
                  </div>
                </div>
              </Tooltip>
              
              {/* Innovation 4 */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-cyan-400 mb-2">🎲 Multi-Phase Training Strategy</div>
                    <p className="mb-2">A carefully designed training pipeline that builds reasoning capabilities progressively through multiple phases.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Phase 1: Pre-training (4.2T tokens)</div>
                      <div className="text-xs text-slate-300">Foundation building with curated high-quality data</div>
                      <div className="text-xs text-slate-300">Focus: Basic reasoning patterns</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Phase 2: Mid-training</div>
                      <div className="text-xs text-slate-300">Limited data with reasoning-focused content</div>
                      <div className="text-xs text-slate-300">Focus: Advanced reasoning skills</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Phase 3: Post-training</div>
                      <div className="text-xs text-slate-300">Supervised fine-tuning and reinforcement learning</div>
                      <div className="text-xs text-slate-300">Focus: Refinement and alignment</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-cyan-400">Progressive complexity</span> throughout training</div>
                      <div className="text-xs text-slate-300">• <span className="text-cyan-400">Specialized phases</span> for different capabilities</div>
                      <div className="text-xs text-slate-300">• <span className="text-cyan-400">Established techniques</span> in post-training</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like learning to walk before you run, then sprint!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-cyan-900/20 to-blue-800/20 backdrop-blur-sm border border-cyan-600/30 rounded-xl p-6 cursor-help hover:border-cyan-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center text-white font-bold">
                      4
                    </div>
                    <h3 className="text-xl font-bold text-white">Multi-Phase Training Strategy</h3>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Progressive training pipeline with pre-training, mid-training, and post-training phases, each optimized for different aspects of reasoning development.
                  </p>
                  <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3">
                    <div className="text-cyan-400 text-sm font-mono">Progressive Complexity = Strong Reasoning</div>
                  </div>
                </div>
              </Tooltip>
            </div>
          </div>

          {/* Training Pipeline */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">🔧</span>
                The Training Pipeline
              </h2>
              <p className="text-slate-400 text-lg">
                A carefully designed multi-phase approach to building reasoning capabilities
              </p>
            </div>
            
            <div className="space-y-6">
              {/* Phase 1: Pre-training */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-blue-400 mb-2">🎯 Phase 1: Pre-training (4.2T tokens)</div>
                    <p className="mb-2">Foundation building with carefully curated high-quality data from multiple sources.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Key Data Sources</div>
                      <div className="text-xs text-slate-300">• FineWeb-Edu (63.75% → 54.83%)</div>
                      <div className="text-xs text-slate-300">• OpenWebMath (6.93% → 23.33%)</div>
                      <div className="text-xs text-slate-300">• StarCoder (10.66% → 0.52%)</div>
                      <div className="text-xs text-slate-300">• Arxiv, StackExchange, Wiki</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Training Details</div>
                      <div className="text-xs text-slate-300">• 2 phases of 2T tokens each</div>
                      <div className="text-xs text-slate-300">• Batch size: 16, Sequence length: 2k</div>
                      <div className="text-xs text-slate-300">• Learning rate: 4e-3 with linear decay</div>
                      <div className="text-xs text-slate-300">• 500k steps per phase</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">Progressive weighting</span> toward reasoning data</div>
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">Quality over quantity</span> approach</div>
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">Cross-domain learning</span> for generalization</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like building a strong foundation before adding specialized skills!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-blue-900/20 to-purple-800/20 backdrop-blur-sm border border-blue-600/30 rounded-xl p-6 cursor-help hover:border-blue-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold">
                      1
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white">Pre-training</h3>
                      <p className="text-slate-400 text-sm">4.2T tokens across 2 phases</p>
                    </div>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Foundation building with curated high-quality data, progressively shifting from general web content to reasoning-specific datasets.
                  </p>
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                    <div className="text-blue-400 text-sm font-mono">Phase 1: General → Phase 2: Reasoning</div>
                  </div>
                </div>
              </Tooltip>
              
              {/* Phase 2: Mid-training */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-purple-400 mb-2">🔄 Phase 2: Mid-training (200B tokens)</div>
                    <p className="mb-2">Knowledge compression and reasoning specialization using data-model co-evolution.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Key Innovation: Data-Model Co-Evolution</div>
                      <div className="text-xs text-slate-300">• Model evaluates its own training data</div>
                      <div className="text-xs text-slate-300">• Removes negative influence samples</div>
                      <div className="text-xs text-slate-300">• Adapts data mixture to model capacity</div>
                      <div className="text-xs text-slate-300">• Self-evolving optimization process</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Training Details</div>
                      <div className="text-xs text-slate-300">• 2 phases of 100B tokens each</div>
                      <div className="text-xs text-slate-300">• Batch size: 4, Sequence length: 4k</div>
                      <div className="text-xs text-slate-300">• Learning rate: 3.6e-4 with linear decay</div>
                      <div className="text-xs text-slate-300">• Knowledge distillation from LLaMA-3.1-8B</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Influence-based filtering</span> of training data</div>
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Adaptive curriculum</span> that grows with model</div>
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Benchmark-free optimization</span></div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like a smart student who knows what to study next!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-purple-900/20 to-indigo-800/20 backdrop-blur-sm border border-purple-600/30 rounded-xl p-6 cursor-help hover:border-purple-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-indigo-600 rounded-lg flex items-center justify-center text-white font-bold">
                      2
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white">Mid-training</h3>
                      <p className="text-slate-400 text-sm">200B tokens with co-evolution</p>
                    </div>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Knowledge compression using data-model co-evolution, where the model actively curates its own training data based on influence scores.
                  </p>
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                    <div className="text-purple-400 text-sm font-mono">Self-Evolving Data Optimization</div>
                  </div>
                </div>
              </Tooltip>
              
              {/* Phase 3: Post-training */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-emerald-400 mb-2">🎯 Phase 3: Post-training</div>
                    <p className="mb-2">Supervised fine-tuning and instruction alignment using established datasets.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Stage 1: General SFT</div>
                      <div className="text-xs text-slate-300">• Tulu-3-SFT (866K samples)</div>
                      <div className="text-xs text-slate-300">• 2 epochs, 4k sequence length</div>
                      <div className="text-xs text-slate-300">• Learning rate: 5e-6</div>
                      <div className="text-xs text-slate-300">• Focus: Instruction following</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Stage 2: Reasoning SFT</div>
                      <div className="text-xs text-slate-300">• OpenMathReasoning (3.2M samples)</div>
                      <div className="text-xs text-slate-300">• OpenScienceReasoning-2 (802K samples)</div>
                      <div className="text-xs text-slate-300">• OpenCodeReasoning-2 (2.2M samples)</div>
                      <div className="text-xs text-slate-300">• 4 epochs, 32k sequence length</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Two-stage approach</span> for better alignment</div>
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Long context reasoning</span> (32k tokens)</div>
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Domain-specific specialization</span></div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like getting specialized training after mastering the basics!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-emerald-900/20 to-teal-800/20 backdrop-blur-sm border border-emerald-600/30 rounded-xl p-6 cursor-help hover:border-emerald-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-lg flex items-center justify-center text-white font-bold">
                      3
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white">Post-training</h3>
                      <p className="text-slate-400 text-sm">SFT with reasoning specialization</p>
                    </div>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Two-stage supervised fine-tuning: first for general instruction following, then for specialized reasoning capabilities with long context support.
                  </p>
                  <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-3">
                    <div className="text-emerald-400 text-sm font-mono">General SFT → Reasoning SFT</div>
                  </div>
                </div>
              </Tooltip>
            </div>
          </div>

          {/* Technical Architecture */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">🏗️</span>
                Technical Architecture
              </h2>
              <p className="text-slate-400 text-lg">
                Model specifications and design choices
              </p>
            </div>
            
            <div className="grid md:grid-cols-3 gap-6 mb-8">
              <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6">
                <div className="text-4xl mb-4 text-center">140M</div>
                <h3 className="text-lg font-bold text-white mb-3 text-center">MobileLLM-R1-140M</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Layers:</span>
                    <span className="text-white">15</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Heads:</span>
                    <span className="text-white">9</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">KV Heads:</span>
                    <span className="text-white">3</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Dim:</span>
                    <span className="text-white">576</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Hidden Dim:</span>
                    <span className="text-white">2048</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6">
                <div className="text-4xl mb-4 text-center">360M</div>
                <h3 className="text-lg font-bold text-white mb-3 text-center">MobileLLM-R1-360M</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Layers:</span>
                    <span className="text-white">15</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Heads:</span>
                    <span className="text-white">16</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">KV Heads:</span>
                    <span className="text-white">4</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Dim:</span>
                    <span className="text-white">1024</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Hidden Dim:</span>
                    <span className="text-white">4096</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6">
                <div className="text-4xl mb-4 text-center">950M</div>
                <h3 className="text-lg font-bold text-white mb-3 text-center">MobileLLM-R1-950M</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Layers:</span>
                    <span className="text-white">22</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Heads:</span>
                    <span className="text-white">24</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">KV Heads:</span>
                    <span className="text-white">6</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Dim:</span>
                    <span className="text-white">1536</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Hidden Dim:</span>
                    <span className="text-white">6144</span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <span>⚙️</span>
                Key Design Features
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <h4 className="text-white font-semibold">LLaMA3.2 Tokenizer</h4>
                      <p className="text-slate-300 text-sm">128k subword vocabulary for efficient tokenization</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-purple-400 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <h4 className="text-white font-semibold">QK-Norm</h4>
                      <p className="text-slate-300 text-sm">Mitigates training instabilities in self-attention blocks</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <h4 className="text-white font-semibold">Weight Sharing</h4>
                      <p className="text-slate-300 text-sm">Input and output embeddings share weights for parameter efficiency</p>
                    </div>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <h4 className="text-white font-semibold">Mobile-Optimized</h4>
                      <p className="text-slate-300 text-sm">Designed specifically for on-device deployment</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <h4 className="text-white font-semibold">Long Context</h4>
                      <p className="text-slate-300 text-sm">Supports up to 32k token context length</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-orange-400 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <h4 className="text-white font-semibold">Reasoning-First</h4>
                      <p className="text-slate-300 text-sm">Architecture optimized for chain-of-thought reasoning</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Results */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">🏆</span>
                The Results
              </h2>
              <p className="text-slate-400 text-lg">
                Remarkable performance with minimal resources
              </p>
            </div>
            
            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <div className="bg-gradient-to-br from-blue-900/20 to-purple-800/20 backdrop-blur-sm border border-blue-600/30 rounded-xl p-6 text-center">
                <div className="text-4xl mb-4">📊</div>
                <h3 className="text-xl font-bold text-white mb-2">AIME Performance</h3>
                <div className="text-blue-400 text-3xl font-bold mb-2">15.5</div>
                <p className="text-slate-300 text-sm mb-3">MobileLLM-R1-950M</p>
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                  <div className="text-blue-400 text-sm">vs 0.6 (OLMo-2-1.48B)</div>
                  <div className="text-blue-400 text-sm">vs 0.3 (SmoILM-2-1.7B)</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-purple-900/20 to-indigo-800/20 backdrop-blur-sm border border-purple-600/30 rounded-xl p-6 text-center">
                <div className="text-4xl mb-4">⚡</div>
                <h3 className="text-xl font-bold text-white mb-2">Efficiency</h3>
                <div className="text-purple-400 text-3xl font-bold mb-2">11.7%</div>
                <p className="text-slate-300 text-sm mb-3">Of Qwen3&apos;s tokens</p>
                <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                  <div className="text-purple-400 text-sm">2T vs 36T tokens</div>
                  <div className="text-purple-400 text-sm">Same or better performance</div>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-emerald-900/20 to-teal-800/20 backdrop-blur-sm border border-emerald-600/30 rounded-xl p-6 mb-8">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <span>📈</span>
                Benchmark Performance Comparison
              </h3>
              <p className="text-slate-300 mb-4">
                MobileLLM-R1-950M matches or surpasses Qwen3-0.6B across multiple reasoning benchmarks despite being trained on significantly less data.
              </p>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-4">
                  <div className="text-emerald-400 font-mono text-sm mb-2">GSM8K (Math)</div>
                  <div className="text-white text-2xl font-bold mb-1">61.6%</div>
                  <div className="text-slate-300 text-xs">vs 60.9% (Qwen3-0.6B)</div>
                </div>
                <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-4">
                  <div className="text-emerald-400 font-mono text-sm mb-2">HumanEval (Code)</div>
                  <div className="text-white text-2xl font-bold mb-1">46.3%</div>
                  <div className="text-slate-300 text-xs">vs 30.5% (Qwen3-0.6B)</div>
                </div>
                <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-4">
                  <div className="text-emerald-400 font-mono text-sm mb-2">MMLU (Knowledge)</div>
                  <div className="text-white text-2xl font-bold mb-1">47.4%</div>
                  <div className="text-slate-300 text-xs">vs 52.4% (Qwen3-0.6B)</div>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <span>📱</span>
                On-Device Performance
              </h3>
              <p className="text-slate-300 mb-4">
                MobileLLM-R1 models demonstrate excellent efficiency for on-device deployment, with the 140M model achieving over 100 tokens/second on mobile devices.
              </p>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-slate-700/30 border border-slate-600/30 rounded-lg p-4 text-center">
                  <div className="text-slate-400 font-mono text-sm mb-2">MobileLLM-R1-140M</div>
                  <div className="text-white text-2xl font-bold mb-1">129.67</div>
                  <div className="text-slate-300 text-xs">tokens/sec (1k context)</div>
                </div>
                <div className="bg-slate-700/30 border border-slate-600/30 rounded-lg p-4 text-center">
                  <div className="text-slate-400 font-mono text-sm mb-2">MobileLLM-R1-360M</div>
                  <div className="text-white text-2xl font-bold mb-1">77.23</div>
                  <div className="text-slate-300 text-xs">tokens/sec (1k context)</div>
                </div>
                <div className="bg-slate-700/30 border border-slate-600/30 rounded-lg p-4 text-center">
                  <div className="text-slate-400 font-mono text-sm mb-2">MobileLLM-R1-950M</div>
                  <div className="text-white text-2xl font-bold mb-1">31.05</div>
                  <div className="text-slate-300 text-xs">tokens/sec (1k context)</div>
                </div>
              </div>
            </div>
          </div>

          {/* Data Curation Methodology */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">🔬</span>
                Data Curation Methodology
              </h2>
              <p className="text-slate-400 text-lg">
                How MobileLLM-R1 identifies and optimizes training data
              </p>
            </div>
            
            <div className="space-y-6">
              {/* Leave-One-Out Analysis */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-blue-400 mb-2">🔍 Leave-One-Out Analysis</div>
                    <p className="mb-2">Systematic evaluation of each dataset&apos;s contribution to reasoning capabilities.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Methodology</div>
                      <div className="text-xs text-slate-300">• Train models excluding one dataset at a time</div>
                      <div className="text-xs text-slate-300">• Measure negative log-likelihood on capability-probing datasets</div>
                      <div className="text-xs text-slate-300">• Quantify each dataset&apos;s impact on reasoning</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Key Findings</div>
                      <div className="text-xs text-slate-300">• FineWeb-Edu: Largest cross-domain benefit</div>
                      <div className="text-xs text-slate-300">• StarCoder: Strong code + math transfer</div>
                      <div className="text-xs text-slate-300">• Math datasets: Primarily benefit math</div>
                      <div className="text-xs text-slate-300">• Wikipedia: Limited reasoning impact</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">Benchmark-free</span> evaluation</div>
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">Cross-domain analysis</span> of data impact</div>
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">Quantitative measurement</span> of contributions</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like testing each ingredient to see which makes the cake taste better!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-blue-900/20 to-purple-800/20 backdrop-blur-sm border border-blue-600/30 rounded-xl p-6 cursor-help hover:border-blue-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold">
                      1
                    </div>
                    <h3 className="text-xl font-bold text-white">Leave-One-Out Analysis</h3>
                  </div>
                    <p className="text-slate-300 mb-3">
                    Systematic evaluation of each dataset&apos;s contribution to reasoning capabilities by training models with and without specific data sources.
                  </p>
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                    <div className="text-blue-400 text-sm font-mono">Quantify Data Impact = Better Mixtures</div>
                  </div>
                </div>
              </Tooltip>
              
              {/* Influence-Based Data Mixing */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-purple-400 mb-2">🎯 Influence-Based Data Mixing</div>
                    <p className="mb-2">Using influence scores to create optimal data mixtures without benchmark exposure.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Influence Score Calculation</div>
                      <div className="text-xs text-slate-300">• Measure impact of each training sample</div>
                      <div className="text-xs text-slate-300">• Cross-capability influence analysis</div>
                      <div className="text-xs text-slate-300">• Self-influence vs cross-influence</div>
                      <div className="text-xs text-slate-300">• Weighted across training checkpoints</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Data Mixture Optimization</div>
                      <div className="text-xs text-slate-300">• Higher influence = higher sampling weight</div>
                      <div className="text-xs text-slate-300">• Cross-domain transfer benefits</div>
                      <div className="text-xs text-slate-300">• Benchmark-free optimization</div>
                      <div className="text-xs text-slate-300">• Self-evolving process</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Principled weighting</span> based on empirical impact</div>
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Cross-domain learning</span> for generalization</div>
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Adaptive optimization</span> throughout training</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like having a smart tutor who knows exactly what to teach next!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-purple-900/20 to-indigo-800/20 backdrop-blur-sm border border-purple-600/30 rounded-xl p-6 cursor-help hover:border-purple-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-indigo-600 rounded-lg flex items-center justify-center text-white font-bold">
                      2
                    </div>
                    <h3 className="text-xl font-bold text-white">Influence-Based Data Mixing</h3>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Principled data mixture optimization using influence scores to weight datasets based on their empirical contribution to reasoning capabilities.
                  </p>
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                    <div className="text-purple-400 text-sm font-mono">Empirical Impact = Optimal Mixtures</div>
                  </div>
                </div>
              </Tooltip>
              
              {/* Hierarchical Rejection Sampling */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-emerald-400 mb-2">🎲 Hierarchical Rejection Sampling</div>
                    <p className="mb-2">Multi-stage filtering to create high-quality capability-probing datasets.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Stage 1: Quality Filtering</div>
                      <div className="text-xs text-slate-300">• FineWeb-Edu classifier (score &gt; 4)</div>
                      <div className="text-xs text-slate-300">• Educational value assessment</div>
                      <div className="text-xs text-slate-300">• Remove low-quality content</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Stage 2: Reasoning Relevance</div>
                      <div className="text-xs text-slate-300">• Ask-LLM paradigm evaluation</div>
                      <div className="text-xs text-slate-300">• Binary classification (include/exclude)</div>
                      <div className="text-xs text-slate-300">• Top 10% confidence samples</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Stage 3: Domain Specialization</div>
                      <div className="text-xs text-slate-300">• Code, Math, Knowledge domains</div>
                      <div className="text-xs text-slate-300">• Domain-specific prompts</div>
                      <div className="text-xs text-slate-300">• Semantic deduplication</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Multi-stage filtering</span> for quality</div>
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Domain-specific optimization</span></div>
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Representative sampling</span> (10K examples)</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like having multiple quality control checkpoints!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-emerald-900/20 to-teal-800/20 backdrop-blur-sm border border-emerald-600/30 rounded-xl p-6 cursor-help hover:border-emerald-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-lg flex items-center justify-center text-white font-bold">
                      3
                    </div>
                    <h3 className="text-xl font-bold text-white">Hierarchical Rejection Sampling</h3>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Multi-stage filtering pipeline combining classifier-based quality assessment with model-based reasoning relevance evaluation.
                  </p>
                  <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-3">
                    <div className="text-emerald-400 text-sm font-mono">Quality + Relevance = Better Data</div>
                  </div>
                </div>
              </Tooltip>
            </div>
          </div>

          {/* Implications */}
          <div className="mb-8">
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                <span className="text-3xl">🚀</span>
                What This Means for AI
              </h2>
              
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">Democratized AI Research</h3>
                    <p className="text-slate-300 text-sm">
                      Small models with strong reasoning capabilities make advanced AI accessible to more researchers and organizations with limited resources.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-purple-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">Mobile and Edge Computing</h3>
                    <p className="text-slate-300 text-sm">
                      Sub-billion parameter models with reasoning capabilities enable sophisticated AI on mobile devices and edge computing platforms.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-emerald-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">Data Efficiency Revolution</h3>
                    <p className="text-slate-300 text-sm">
                      Proves that smart data curation can achieve better results than massive datasets, reducing computational and environmental costs.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">Reasoning-First Design</h3>
                    <p className="text-slate-300 text-sm">
                      Shifts focus from scale to capability, encouraging research into reasoning-specific architectures and training methods.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">Open Research</h3>
                    <p className="text-slate-300 text-sm">
                      Complete training recipes, data sources, and model checkpoints are released, enabling reproducible research and further innovation.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Resources */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-4">
              📚 Learn More
            </h2>
            
            <div className="flex flex-col sm:flex-row gap-4">
              <a 
                href="https://arxiv.org/pdf/2509.24945"
                target="_blank"
                rel="noopener noreferrer"
                className="group inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-blue-500/25"
              >
                <svg className="w-5 h-5 group-hover:rotate-12 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span className="group-hover:translate-x-1 transition-transform duration-300">
                  Read Paper
                </span>
              </a>
              
              <a 
                href="https://github.com/facebookresearch/MobileLLM-R1"
                target="_blank"
                rel="noopener noreferrer"
                className="group inline-flex items-center gap-2 px-6 py-3 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-blue-500 hover:text-blue-400 transition-all duration-300 hover:bg-blue-500/10"
              >
                <svg className="w-5 h-5 group-hover:rotate-12 transition-transform duration-300" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                <span className="group-hover:translate-x-1 transition-transform duration-300">
                  View Code
                </span>
              </a>

              <a 
                href="https://huggingface.co/collections/facebook/mobilellm-r1-68c4597b104fac45f28f448e"
                target="_blank"
                rel="noopener noreferrer"
                className="group inline-flex items-center gap-2 px-6 py-3 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-purple-500 hover:text-purple-400 transition-all duration-300 hover:bg-purple-500/10"
              >
                <svg className="w-5 h-5 group-hover:rotate-12 transition-transform duration-300" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm0 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10z"/>
                  <path d="M12 6c-3.314 0-6 2.686-6 6s2.686 6 6 6 6-2.686 6-6-2.686-6-6-6zm0 10c-2.206 0-4-1.794-4-4s1.794-4 4-4 4 1.794 4 4-1.794 4-4 4z"/>
                </svg>
                <span className="group-hover:translate-x-1 transition-transform duration-300">
                  HuggingFace Models
                </span>
              </a>
            </div>
          </div>

          {/* Back to Home */}
          <div className="text-center">
            <Link 
              href="/"
              className="inline-flex items-center gap-2 px-6 py-3 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-blue-500 hover:text-blue-400 transition-all duration-300 transform hover:scale-105"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to Home
            </Link>
          </div>
        </div>
      </main>
    </>
  );
}
