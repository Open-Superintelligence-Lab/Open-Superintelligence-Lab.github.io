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

export default function NVFP4Project() {
  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-transparent"></div>
        </div>
        
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/6 left-1/6 w-3 h-3 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full opacity-60 animate-pulse"></div>
          <div className="absolute top-1/4 right-1/5 w-2 h-2 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full opacity-50 animate-pulse delay-300"></div>
          <div className="absolute top-1/3 left-1/8 w-4 h-4 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full opacity-40 animate-pulse delay-700"></div>
          <div className="absolute bottom-1/4 right-1/6 w-2.5 h-2.5 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full opacity-55 animate-pulse delay-1000"></div>
        </div>
        
        <div className="relative container mx-auto px-6 pt-32 pb-24">
          <div className="text-center max-w-4xl mx-auto">
            <div className="relative">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium mb-8 leading-tight">
                <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                  NVIDIA's 4-Bit Revolution
                </span>
              </h1>
              <div className="text-lg md:text-xl text-slate-400 mb-4">
                ⚡ NVFP4: 2-3x Faster Training, 50% Less Memory
              </div>
              
              <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-medium leading-tight blur-sm">
                <span className="bg-gradient-to-r from-blue-400/20 via-purple-400/20 to-cyan-400/20 bg-clip-text text-transparent">
                  NVIDIA's 4-Bit Revolution
                </span>
              </div>
            </div>
            
            <p className="text-xl text-slate-300 mb-12 leading-relaxed">
              How NVIDIA trained a 12B parameter hybrid Mamba-Transformer model on 10 trillion tokens using 4-bit precision without losing performance
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
        <div className="container mx-auto px-6 max-w-4xl">
          
          {/* TL;DR Section */}
          <div className="mb-8">
            <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 backdrop-blur-sm border border-blue-600/30 rounded-xl p-6">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center gap-3">
                <span className="text-4xl">📝</span>
                TL;DR
              </h2>
              <p className="text-slate-300 leading-relaxed mb-4">
                NVIDIA has figured out how to train massive LLMs using a new <strong className="text-blue-400">4-bit number format called NVFP4</strong>, which is a huge deal for efficiency. Training in 4-bit is much faster and uses less memory than the current 8-bit standard (FP8), but it's very difficult to do without the model's performance collapsing.
              </p>
              <p className="text-slate-300 leading-relaxed">
                Their solution combines four key techniques to train a <strong className="text-purple-400">12-billion-parameter hybrid Mamba-Transformer model on 10 trillion tokens</strong> with performance nearly identical to FP8 training. This marks the first successful demonstration of training billion-parameter language models with 4-bit precision over a multi-trillion-token horizon.
              </p>
            </div>
          </div>

          {/* The Problem */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">⚠️</span>
                The Challenge: Why 4-Bit is Hard
              </h2>
              <p className="text-slate-400 text-lg">
                The cost of AI training is exploding
              </p>
            </div>
            
            <div className="grid md:grid-cols-3 gap-6 mb-8">
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-orange-400 mb-2">🔢 Current Standard: FP8</div>
                    <p className="mb-2">8-bit floating point (FP8) is the current industry standard for efficient LLM training.</p>
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• 256 possible values (2⁸)</div>
                      <div className="text-xs text-slate-300">• Good precision</div>
                      <div className="text-xs text-slate-300">• Moderate speed</div>
                    </div>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-orange-900/20 to-orange-800/20 backdrop-blur-sm border border-orange-600/30 rounded-xl p-6 text-center cursor-help hover:border-orange-500/50 transition-all duration-300">
                  <div className="text-4xl mb-4">🔢</div>
                  <h3 className="text-xl font-bold text-white mb-2">FP8 (Current)</h3>
                  <div className="text-orange-400 font-mono text-2xl mb-2">256 values</div>
                  <p className="text-slate-300 text-sm">8-bit precision</p>
                </div>
              </Tooltip>
              
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-blue-400 mb-2">✨ New Format: NVFP4</div>
                    <p className="mb-2">4-bit floating point has only 16 possible values, making it extremely challenging but highly efficient.</p>
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• Only 16 possible values (2⁴)</div>
                      <div className="text-xs text-slate-300">• 2-3x faster compute</div>
                      <div className="text-xs text-slate-300">• 50% less memory</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">The key challenge: representing numbers accurately with so few values!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 backdrop-blur-sm border border-blue-600/30 rounded-xl p-6 text-center cursor-help hover:border-blue-500/50 transition-all duration-300">
                  <div className="text-4xl mb-4">✨</div>
                  <h3 className="text-xl font-bold text-white mb-2">NVFP4 (New!)</h3>
                  <div className="text-blue-400 font-mono text-2xl mb-2">16 values</div>
                  <p className="text-slate-300 text-sm">4-bit precision</p>
                </div>
              </Tooltip>
              
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-emerald-400 mb-2">📊 The Benefits</div>
                    <p className="mb-2">NVFP4 enables dramatic improvements in training efficiency.</p>
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• 2-3x faster calculations</div>
                      <div className="text-xs text-slate-300">• 50% memory reduction</div>
                      <div className="text-xs text-slate-300">• Same model quality</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">This means faster, cheaper, and more energy-efficient AI!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-emerald-900/20 to-emerald-800/20 backdrop-blur-sm border border-emerald-600/30 rounded-xl p-6 text-center cursor-help hover:border-emerald-500/50 transition-all duration-300">
                  <div className="text-4xl mb-4">🚀</div>
                  <h3 className="text-xl font-bold text-white mb-2">Performance</h3>
                  <div className="text-emerald-400 font-mono text-2xl mb-2">2-3x faster</div>
                  <p className="text-slate-300 text-sm">50% less memory</p>
                </div>
              </Tooltip>
            </div>
          </div>

          {/* NVFP4 Format Comparison */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">🔬</span>
                NVFP4 vs MXFP4
              </h2>
              <p className="text-slate-400 text-lg">
                How NVIDIA's format improves on the standard
              </p>
            </div>
            
            <div className="space-y-4">
              {/* Block Size Comparison */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-slate-300 mb-2">📦 Block Size Impact</div>
                    <p className="mb-2">Block size determines how many numbers share a single scale factor.</p>
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-orange-400">MXFP4</span>: 32 numbers per block</div>
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">NVFP4</span>: 16 numbers per block</div>
                    </div>
                    <p className="text-xs text-slate-400">Smaller blocks = less variation = better scale factor fit = more accurate quantization!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6 hover:border-slate-500/50 transition-all duration-300">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">Block Size</h3>
                    <div className="flex items-center gap-4">
                      <div className="text-center">
                        <div className="text-orange-400 font-mono text-sm">32 numbers</div>
                        <div className="text-slate-400 text-xs">MXFP4</div>
                      </div>
                      <div className="text-slate-400">→</div>
                      <div className="text-center">
                        <div className="text-blue-400 font-mono text-sm">16 numbers</div>
                        <div className="text-slate-400 text-xs">NVFP4</div>
                      </div>
                    </div>
                  </div>
                  <p className="text-slate-300 text-sm">Smaller blocks = better fit</p>
                </div>
              </Tooltip>

              {/* Scale Format Comparison */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-slate-300 mb-2">🎯 Scale Format Precision</div>
                    <p className="mb-2">Scale format determines how precisely we can represent scale factors.</p>
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-orange-400">UE8M0</span>: Power-of-two only (2, 4, 8, 16...)</div>
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">E4M3</span>: More precise (2.5, 4.75, 8.25...)</div>
                    </div>
                    <p className="text-xs text-slate-400">More precise scaling = less rounding error = better preservation of information!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6 hover:border-slate-500/50 transition-all duration-300">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">Scale Format</h3>
                    <div className="flex items-center gap-4">
                      <div className="text-center">
                        <div className="text-orange-400 font-mono text-sm">UE8M0 (crude)</div>
                        <div className="text-slate-400 text-xs">MXFP4</div>
                      </div>
                      <div className="text-slate-400">→</div>
                      <div className="text-center">
                        <div className="text-blue-400 font-mono text-sm">E4M3 (precise)</div>
                        <div className="text-slate-400 text-xs">NVFP4</div>
                      </div>
                    </div>
                  </div>
                  <p className="text-slate-300 text-sm">More accurate scaling</p>
                </div>
              </Tooltip>

              {/* Scaling Strategy Comparison */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-slate-300 mb-2">📊 Two-Level Scaling Strategy</div>
                    <p className="mb-2">NVFP4 uses a sophisticated two-level scaling approach for maximum flexibility.</p>
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-orange-400">Single-level</span>: One scale per block</div>
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">Two-level</span>: Tensor-wide + per-block scales</div>
                    </div>
                    <p className="text-xs text-slate-400">Like adjusting overall brightness (tensor) then fine-tuning contrast (blocks) for perfect representation!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-slate-800/30 to-slate-700/30 backdrop-blur-sm border border-slate-600/30 rounded-xl p-6 hover:border-slate-500/50 transition-all duration-300">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">Scaling Strategy</h3>
                    <div className="flex items-center gap-4">
                      <div className="text-center">
                        <div className="text-orange-400 font-mono text-sm">Single-level</div>
                        <div className="text-slate-400 text-xs">MXFP4</div>
                      </div>
                      <div className="text-slate-400">→</div>
                      <div className="text-center">
                        <div className="text-blue-400 font-mono text-sm">Two-level</div>
                        <div className="text-slate-400 text-xs">NVFP4</div>
                      </div>
                    </div>
                  </div>
                  <p className="text-slate-300 text-sm">Better dynamic range</p>
                </div>
              </Tooltip>
            </div>
          </div>

          {/* The 4 Key Techniques */}
          <div className="mb-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <span className="text-4xl">🔑</span>
                The 4 Key Techniques
              </h2>
              <p className="text-slate-400 text-lg">
                The "secret sauce" that makes NVFP4 work
              </p>
            </div>
            
            <div className="space-y-6">
              {/* Technique 1 */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-purple-400 mb-2">🎯 Mixed Precision Strategy</div>
                    <p className="mb-2">Some layers are more numerically sensitive than others, especially at the beginning and end of the network.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Layer Sensitivity Analysis</div>
                      <div className="text-xs text-slate-300">Input embedding: <span className="text-red-400">Very sensitive</span> (BF16)</div>
                      <div className="text-xs text-slate-300">Hidden layers 1-20: <span className="text-green-400">Robust</span> (NVFP4)</div>
                      <div className="text-xs text-slate-300">Output head: <span className="text-red-400">Very sensitive</span> (BF16)</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Precision Distribution (12B Model)</div>
                      <div className="text-xs text-slate-300">BF16 layers: First 2 + last 8 blocks (16% of linear layers)</div>
                      <div className="text-xs text-slate-300">NVFP4 layers: Middle 52 blocks (84% of linear layers)</div>
                      <div className="text-xs text-slate-300">Memory savings: ~42% overall</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Input/Output</span>: Critical for accuracy</div>
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Middle layers</span>: Can tolerate quantization</div>
                      <div className="text-xs text-slate-300">• <span className="text-purple-400">Result</span>: Best of both worlds</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like using premium materials for the foundation and roof, standard for the walls!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-purple-900/20 to-purple-800/20 backdrop-blur-sm border border-purple-600/30 rounded-xl p-6 cursor-help hover:border-purple-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold">
                      1
                    </div>
                    <h3 className="text-xl font-bold text-white">Selective High-Precision Layers</h3>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Keep sensitive layers (first/last ~15%) in higher precision (BF16), while using NVFP4 for the bulk of computation.
                  </p>
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                    <div className="text-purple-400 text-sm font-mono">~15% BF16 + ~85% NVFP4 = Stable Training</div>
                  </div>
                </div>
              </Tooltip>
              
              {/* Technique 2 */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-blue-400 mb-2">🔄 Random Hadamard Transform (RHT)</div>
                    <p className="mb-2">Outliers (extreme values) force all other values to be crushed near zero when quantized.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Example: Before RHT</div>
                      <div className="text-xs text-slate-300">Values: [0.1, 0.2, 0.3, <span className="text-red-400 font-bold">15.7</span>, 0.4, 0.5]</div>
                      <div className="text-xs text-slate-300">Scale factor: 15.7 (dominated by outlier)</div>
                      <div className="text-xs text-slate-300">Quantized: [0, 0, 0, <span className="text-red-400">15</span>, 0, 0]</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">After RHT (simplified)</div>
                      <div className="text-xs text-slate-300">Values: [2.1, 2.3, 2.5, 2.7, 2.9, 3.1]</div>
                      <div className="text-xs text-slate-300">Scale factor: 3.1 (much more reasonable)</div>
                      <div className="text-xs text-slate-300">Quantized: [2, 2, 3, 3, 3, 3]</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">RHT</span>: H = (1/√2) × H₂ ⊗ H<sub>d/2</sub> with random sign vector</div>
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">Matrix size</span>: 16×16 optimal for 12B models</div>
                      <div className="text-xs text-slate-300">• <span className="text-blue-400">Applied to</span>: Wgrad inputs only (not Fprop/Dgrad)</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like spreading butter evenly instead of having lumps - all values get fair representation!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 backdrop-blur-sm border border-blue-600/30 rounded-xl p-6 cursor-help hover:border-blue-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center text-white font-bold">
                      2
                    </div>
                    <h3 className="text-xl font-bold text-white">Random Hadamard Transforms (RHT)</h3>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Mathematical operation that "smears" extreme outlier values across all values, making distributions more uniform and easier to quantize.
                  </p>
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                    <div className="text-blue-400 text-sm font-mono">Outliers → Uniform Distribution</div>
                  </div>
                </div>
              </Tooltip>
              
              {/* Technique 3 */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-emerald-400 mb-2">📐 2D Scaling Consistency</div>
                    <p className="mb-2">In backpropagation, weight matrices are transposed. Row-wise scaling becomes column-wise, breaking consistency.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Example: 1D Scaling Problem</div>
                      <div className="text-xs text-slate-300">Forward: W = [1, 2, 3] → scaled row-wise</div>
                      <div className="text-xs text-slate-300">Backward: W^T = [1, 2, 3] → scaled column-wise</div>
                      <div className="text-xs text-red-400">❌ Different scaling = broken chain rule!</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">2D Block Solution</div>
                      <div className="text-xs text-slate-300">16×16 block: [[1,2,3,4], [5,6,7,8], ...]</div>
                      <div className="text-xs text-slate-300">Same block scaling for W and W^T</div>
                      <div className="text-xs text-emerald-400">✅ Transpose-invariant = consistent!</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Forward</span>: W scaled in 16×16 blocks</div>
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Backward</span>: W^T scaled in same 16×16 blocks</div>
                      <div className="text-xs text-slate-300">• <span className="text-emerald-400">Result</span>: Chain rule preserved</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like having the same ruler for measuring in both directions!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-emerald-900/20 to-emerald-800/20 backdrop-blur-sm border border-emerald-600/30 rounded-xl p-6 cursor-help hover:border-emerald-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-lg flex items-center justify-center text-white font-bold">
                      3
                    </div>
                    <h3 className="text-xl font-bold text-white">Two-Dimensional (2D) Scaling</h3>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Scale weights in 16×16 2D blocks instead of 1D rows, ensuring consistency between forward and backward passes when matrices are transposed.
                  </p>
                  <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-3">
                    <div className="text-emerald-400 text-sm font-mono">Forward ≡ Backward (Transpose-Invariant)</div>
                  </div>
                </div>
              </Tooltip>
              
              {/* Technique 4 */}
              <Tooltip 
                content={
                  <div>
                    <div className="font-bold text-cyan-400 mb-2">🎲 Stochastic Rounding</div>
                    <p className="mb-2">Standard rounding introduces systematic bias that accumulates over billions of operations.</p>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Example: Value 2.7</div>
                      <div className="text-xs text-slate-300">Deterministic: 2.7 → 3 (always)</div>
                      <div className="text-xs text-slate-300">Stochastic: 2.7 → 3 (70%) or 2 (30%)</div>
                      <div className="text-xs text-slate-300">Expected: 0.7×3 + 0.3×2 = 2.7 ✅</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300 font-semibold mb-1">Bias Accumulation Problem</div>
                      <div className="text-xs text-slate-300">1M operations: 2.7 → 3 (always)</div>
                      <div className="text-xs text-slate-300">Total bias: 1M × 0.3 = 300,000</div>
                      <div className="text-xs text-red-400">❌ Systematic error grows!</div>
                    </div>
                    
                    <div className="bg-slate-700/50 rounded p-2 mb-2">
                      <div className="text-xs text-slate-300">• <span className="text-cyan-400">Formula</span>: P(round up) = x - floor(x)</div>
                      <div className="text-xs text-slate-300">• <span className="text-cyan-400">Applied to</span>: Gradients only (not weights/activations)</div>
                      <div className="text-xs text-slate-300">• <span className="text-cyan-400">Result</span>: Unbiased on average, prevents divergence</div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">Like flipping a weighted coin - fair in the long run!</p>
                  </div>
                }
              >
                <div className="bg-gradient-to-br from-cyan-900/20 to-cyan-800/20 backdrop-blur-sm border border-cyan-600/30 rounded-xl p-6 cursor-help hover:border-cyan-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-cyan-600 rounded-lg flex items-center justify-center text-white font-bold">
                      4
                    </div>
                    <h3 className="text-xl font-bold text-white">Stochastic Rounding</h3>
                  </div>
                  <p className="text-slate-300 mb-3">
                    Probabilistic rounding instead of deterministic "round-to-nearest" eliminates systematic bias that accumulates in gradient calculations.
                  </p>
                  <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3">
                    <div className="text-cyan-400 text-sm font-mono">Unbiased Gradients = Better Training</div>
                  </div>
                </div>
              </Tooltip>
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
                Massive efficiency gains with minimal performance loss
              </p>
            </div>
            
            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <div className="bg-gradient-to-br from-green-900/20 to-green-800/20 backdrop-blur-sm border border-green-600/30 rounded-xl p-6 text-center">
                <div className="text-4xl mb-4">📊</div>
                <h3 className="text-xl font-bold text-white mb-2">Training Success</h3>
                <div className="text-green-400 text-3xl font-bold mb-2">12B params</div>
                <p className="text-slate-300 text-sm mb-3">10 trillion tokens trained</p>
                <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                  <div className="text-green-400 text-sm">Hybrid Mamba-Transformer architecture</div>
                  <div className="text-green-400 text-sm">First multi-trillion-token 4-bit training</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 backdrop-blur-sm border border-blue-600/30 rounded-xl p-6 text-center">
                <div className="text-4xl mb-4">⚡</div>
                <h3 className="text-xl font-bold text-white mb-2">Performance Match</h3>
                <div className="text-blue-400 text-3xl font-bold mb-2">~99%</div>
                <p className="text-slate-300 text-sm mb-3">Of FP8 baseline performance</p>
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                  <div className="text-blue-400 text-sm">MMLU-pro: 62.58% vs 62.62% (FP8)</div>
                  <div className="text-blue-400 text-sm">Math: 86.88% vs 86.20% (FP8)</div>
                  <div className="text-blue-400 text-sm">GSM8k: 92.27% vs 89.08% (FP8)</div>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-purple-900/20 to-purple-800/20 backdrop-blur-sm border border-purple-600/30 rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <span>📈</span>
                NVFP4 vs MXFP4
              </h3>
              <p className="text-slate-300 mb-4">
                In direct comparison on an 8B model, MXFP4 needed <strong className="text-purple-400">36% more training data</strong> (1.36T vs 1T tokens) to match NVFP4's performance. This proves NVFP4's superior design.
              </p>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                  <div className="text-purple-400 font-mono text-sm mb-1">NVFP4</div>
                  <div className="text-slate-300 text-xs">Better accuracy with same data</div>
                  <div className="text-slate-300 text-xs">E4M3 scale factors, 16-element blocks</div>
                </div>
                <div className="bg-slate-700/30 border border-slate-600/30 rounded-lg p-4">
                  <div className="text-slate-400 font-mono text-sm mb-1">MXFP4</div>
                  <div className="text-slate-300 text-xs">Needs 36% more data to catch up</div>
                  <div className="text-slate-300 text-xs">UE8M0 scale factors, 32-element blocks</div>
                </div>
              </div>
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
                    <h3 className="text-lg font-semibold text-white mb-1">Faster Training</h3>
                    <p className="text-slate-300 text-sm">
                      2-3x speedup means experiments that took weeks now take days. Faster iteration = faster progress.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-purple-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">Lower Cost</h3>
                    <p className="text-slate-300 text-sm">
                      50% memory reduction means you can train larger models on the same hardware, or the same model at half the cost.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-emerald-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">More Accessible AI</h3>
                    <p className="text-slate-300 text-sm">
                      Democratizes AI research by reducing computational barriers. More researchers can train frontier models.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">Green AI</h3>
                    <p className="text-slate-300 text-sm">
                      Massive reduction in energy consumption for training makes AI more sustainable and environmentally friendly.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">Blackwell GPU Ready</h3>
                    <p className="text-slate-300 text-sm">
                      Native Tensor Core support for NVFP4 on NVIDIA Blackwell GPUs delivers 4× speedup on GB200 and 6× on GB300 chips.
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
                href="https://arxiv.org/pdf/2509.25149"
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
                href="https://github.com/NVIDIA/TransformerEngine/pull/2177/files"
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
