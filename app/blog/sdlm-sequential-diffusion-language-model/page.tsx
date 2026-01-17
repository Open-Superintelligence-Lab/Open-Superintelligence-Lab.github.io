import Link from "next/link";

export default function SDLMPage() {

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Background effects */}
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 via-blue-600/20 to-cyan-600/20"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-purple-500/5 to-transparent"></div>
        </div>

        {/* Animated background particles */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/6 left-1/6 w-3 h-3 bg-gradient-to-r from-purple-400 to-blue-400 rounded-full opacity-60 animate-pulse"></div>
          <div className="absolute top-1/4 right-1/5 w-2 h-2 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full opacity-50 animate-pulse delay-300"></div>
          <div className="absolute top-1/3 left-1/8 w-4 h-4 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full opacity-40 animate-pulse delay-700"></div>
          <div className="absolute bottom-1/4 right-1/6 w-2.5 h-2.5 bg-gradient-to-r from-purple-400 to-blue-400 rounded-full opacity-55 animate-pulse delay-1000"></div>
          <div className="absolute bottom-1/3 left-1/4 w-1.5 h-1.5 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full opacity-60 animate-pulse delay-500"></div>
          <div className="absolute top-2/3 right-1/3 w-3.5 h-3.5 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full opacity-35 animate-pulse delay-1200"></div>
        </div>

        <div className="relative container mx-auto px-6 pt-32 pb-24">
          <div className="text-center max-w-5xl mx-auto">
            <div className="relative">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium mb-8 leading-tight">
                <span className="bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  SDLM
                </span>
              </h1>
              <h2 className="text-2xl md:text-3xl lg:text-4xl font-medium mb-6 leading-tight text-slate-300">
                Sequential Diffusion Language Model
              </h2>

              {/* Glow effect for the title */}
              <div className="absolute inset-0 text-4xl md:text-5xl lg:text-6xl font-medium leading-tight blur-sm">
                <span className="bg-gradient-to-r from-purple-400/20 via-blue-400/20 to-cyan-400/20 bg-clip-text text-transparent">
                  SDLM
                </span>
              </div>
            </div>

            <p className="text-xl text-slate-300 mb-8 leading-relaxed max-w-4xl mx-auto">
              Enhances pre-trained autoregressive language models by adaptively determining generation length and maintaining KV-cache compatibility, achieving high efficiency and throughput.
            </p>

            {/* Key Features */}
            <div className="flex flex-wrap justify-center gap-4 text-sm text-slate-400 mb-12">
              <span className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-full">
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
                2x Faster
              </span>
              <span className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-full">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse delay-300"></div>
                KV-Cache Compatible
              </span>
              <span className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-full">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse delay-700"></div>
                High Performance
              </span>
            </div>

            {/* Call to action buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <a
                href="https://github.com/OpenGVLab/SDLM"
                target="_blank"
                rel="noopener noreferrer"
                className="group px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-xl hover:from-purple-700 hover:to-blue-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-purple-500/25"
              >
                <span className="flex items-center gap-2">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                  </svg>
                  View on GitHub
                </span>
              </a>
              <a
                href="https://arxiv.org/pdf/2509.24007"
                target="_blank"
                rel="noopener noreferrer"
                className="group px-8 py-4 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-purple-500 hover:text-purple-400 transition-all duration-300 transform hover:scale-105"
              >
                <span className="flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                  Read Paper
                </span>
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12">
        <div className="container mx-auto px-6 max-w-6xl">

          {/* Introduction Section */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">
              Introduction
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              {/* Autoregression */}
              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-red-500 to-pink-500 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Autoregression
                </h3>
                <p className="text-slate-300 text-sm">
                  Predicts tokens one by one
                </p>
              </div>

              {/* Diffusion */}
              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-yellow-500 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Diffusion
                </h3>
                <p className="text-slate-300 text-sm">
                  Regenerates all tokens each step
                </p>
              </div>

              {/* SDLM */}
              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-blue-500 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  SDLM (Ours)
                </h3>
                <p className="text-slate-300 text-sm">
                  Decodes D tokens per step, keeps confident tokens
                </p>
              </div>
            </div>

            <p className="text-slate-300 leading-relaxed">
              SDLM delivers strong performance with significantly faster decoding speed. It operates approximately 2x faster than comparable autoregressive models while matching their accuracy, and achieves up to 5x speedup over other diffusion language models, as evidenced by results on the MATH-500 benchmark.
            </p>
          </div>

          {/* Model Zoo Section */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">
              Model Zoo
            </h2>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-600/50">
                    <th className="text-left py-3 px-4 text-slate-300 font-semibold">
                      Model Name
                    </th>
                    <th className="text-left py-3 px-4 text-slate-300 font-semibold">
                      Base Model
                    </th>
                    <th className="text-left py-3 px-4 text-slate-300 font-semibold">
                      HuggingFace Link
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-slate-600/30">
                    <td className="py-3 px-4 text-white font-medium">SDLM-3B-D4</td>
                    <td className="py-3 px-4 text-slate-300">Qwen2.5-3B</td>
                    <td className="py-3 px-4">
                      <a
                        href="https://huggingface.co/OpenGVLab/SDLM-3B-D4"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-purple-400 hover:text-purple-300 transition-colors"
                      >
                        OpenGVLab/SDLM-3B-D4
                      </a>
                    </td>
                  </tr>
                  <tr className="border-b border-slate-600/30">
                    <td className="py-3 px-4 text-white font-medium">SDLM-3B-D8</td>
                    <td className="py-3 px-4 text-slate-300">Qwen2.5-3B</td>
                    <td className="py-3 px-4">
                      <a
                        href="https://huggingface.co/OpenGVLab/SDLM-3B-D8"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-purple-400 hover:text-purple-300 transition-colors"
                      >
                        OpenGVLab/SDLM-3B-D8
                      </a>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 text-white font-medium">SDLM-32B-D4</td>
                    <td className="py-3 px-4 text-slate-300">Qwen2.5-32B</td>
                    <td className="py-3 px-4">
                      <a
                        href="https://huggingface.co/OpenGVLab/SDLM-32B-D4"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-purple-400 hover:text-purple-300 transition-colors"
                      >
                        OpenGVLab/SDLM-32B-D4
                      </a>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Performance Section */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">
              Performance
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-white">92.4</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">GSM8K</h3>
                <p className="text-slate-300 text-sm">SDLM-32B</p>
              </div>

              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-white">74.2</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">MATH</h3>
                <p className="text-slate-300 text-sm">SDLM-32B</p>
              </div>

              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-white">81.1</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">HumanEval</h3>
                <p className="text-slate-300 text-sm">SDLM-32B</p>
              </div>

              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-yellow-500 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-white">80.9</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">MBPP</h3>
                <p className="text-slate-300 text-sm">SDLM-32B</p>
              </div>
            </div>

            <div className="bg-slate-800/30 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">
                Efficiency Gains
              </h3>
              <ul className="space-y-2 text-slate-300">
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  Each forward pass generates ~2 tokens on average
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                  ≈2× speedup over autoregressive models
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  Two-thirds latency of AR models
                </li>
              </ul>
            </div>
          </div>

          {/* Methods Section */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">
              Methods
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-lg font-semibold text-white mb-4">
                  Training Pipeline
                </h3>
                <p className="text-slate-300 mb-4 leading-relaxed">
                  The reordered input sequence enables structured masking with causal prefix, visible cross-block prefix, and intra-block bidirectional attention.
                </p>
                <ul className="space-y-2 text-slate-300 text-sm">
                  <li className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                    Causal prefix (top-left)
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    Visible cross-block prefix (bottom-left)
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
                    Intra-block bidirectional attention (bottom-right)
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-white mb-4">
                  Sampling Pipeline
                </h3>
                <p className="text-slate-300 mb-4 leading-relaxed">
                  Confidence-based dynamic block decoding with KV cache reuse. At each step, a block of D tokens is predicted with D-1 padding masks. The longest high-confidence prefix is selected as dynamic output.
                </p>
                <ul className="space-y-2 text-slate-300 text-sm">
                  <li className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    Dynamic block decoding
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
                    KV cache reuse
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-pink-400 rounded-full"></div>
                    Confidence-based selection
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* Inference Section */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">
              Inference
            </h2>

            <div className="bg-slate-900/50 rounded-lg p-6 mb-6">
              <h3 className="text-lg font-semibold text-white mb-4">
                Quick Start with HuggingFace
              </h3>
              <pre className="text-sm text-slate-300 overflow-x-auto">
                <code>{`import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sdlm_inference import SDLM_generate

# Load model and tokenizer
ckpt_hf = 'OpenGVLab/SDLM-3B-D4'
model = AutoModelForCausalLM.from_pretrained(
    ckpt_hf, 
    attn_implementation="eager",
    trust_remote_code=True
).to(dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(ckpt_hf)

# Prepare input
prompt = 'Write a Fibonacci function in Python.'
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Generate response
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
response, history = SDLM_generate(
    model,
    tokenizer,
    model_inputs,
    max_gen_len=1024,
    temperature=0,
    threshold=0.5,
    n_future_tokens=4,
    alg='prob_conf',
    save_history=True,
    use_cache=True
)

print('response: ', response[0])`}</code>
              </pre>
            </div>
          </div>

          {/* Training Section */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">
              Training
            </h2>

            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-4">
                  Environment Setup
                </h3>
                <div className="bg-slate-900/50 rounded-lg p-4">
                  <pre className="text-sm text-slate-300">
                    <code>{`git clone https://github.com/OpenGVLab/SDLM.git
cd SDLM`}</code>
                  </pre>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-white mb-4">
                  Key Dependencies
                </h3>
                <div className="bg-slate-900/50 rounded-lg p-4">
                  <pre className="text-sm text-slate-300">
                    <code>{`transformers==4.37.2
deepspeed==0.16.5
torch>=2.5.0
accelerate==0.32.1`}</code>
                  </pre>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-white mb-4">
                  Training Dataset
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-600/50">
                        <th className="text-left py-3 px-4 text-slate-300 font-semibold">
                          Dataset Name
                        </th>
                        <th className="text-left py-3 px-4 text-slate-300 font-semibold">
                          Samples
                        </th>
                        <th className="text-left py-3 px-4 text-slate-300 font-semibold">
                          Domain
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-slate-600/30">
                        <td className="py-3 px-4 text-slate-300">ScaleQuest-Math</td>
                        <td className="py-3 px-4 text-slate-300">1,000K</td>
                        <td className="py-3 px-4 text-slate-300">Math</td>
                      </tr>
                      <tr className="border-b border-slate-600/30">
                        <td className="py-3 px-4 text-slate-300">Opc-sft-stage2</td>
                        <td className="py-3 px-4 text-slate-300">436K</td>
                        <td className="py-3 px-4 text-slate-300">Code</td>
                      </tr>
                      <tr className="border-b border-slate-600/30">
                        <td className="py-3 px-4 text-slate-300">Smoltalk</td>
                        <td className="py-3 px-4 text-slate-300">1,100K</td>
                        <td className="py-3 px-4 text-slate-300">General</td>
                      </tr>
                      <tr className="border-b border-slate-600/30">
                        <td className="py-3 px-4 text-slate-300">Tulu-3-sft-mixture</td>
                        <td className="py-3 px-4 text-slate-300">939K</td>
                        <td className="py-3 px-4 text-slate-300">General</td>
                      </tr>
                      <tr className="border-b border-slate-600/30">
                        <td className="py-3 px-4 text-slate-300">SciRIFF</td>
                        <td className="py-3 px-4 text-slate-300">79K</td>
                        <td className="py-3 px-4 text-slate-300">Science</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 text-slate-300">Table-GPT</td>
                        <td className="py-3 px-4 text-slate-300">13K</td>
                        <td className="py-3 px-4 text-slate-300">Table</td>
                      </tr>
                    </tbody>
                    <tfoot>
                      <tr className="border-t border-slate-600/50">
                        <td className="py-3 px-4 text-white font-semibold">Total</td>
                        <td className="py-3 px-4 text-white font-semibold">3,506K</td>
                        <td className="py-3 px-4 text-slate-300">--</td>
                      </tr>
                    </tfoot>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Citation Section */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">
              Citation
            </h2>

            <div className="bg-slate-900/50 rounded-lg p-6">
              <pre className="text-sm text-slate-300 overflow-x-auto">
                <code>{`@article{liu2025sdlm,
  title={Sequential Diffusion Language Models},
  author={Liu, Yangzhou and Cao, Yue and Li, Hao and Luo, Gen and Chen, Zhe and Wang, Weiyun and Liang, Xiaobo and Qi, Biqing and Wu, Lijun and Tian, Changyao and Zhang, Yanting and Li, Yuqiang and Lu, Tong and Qiao, Yu and Dai, Jifeng and Wang, Wenhai},
  journal={arXiv preprint arXiv:2509.24007},
  year={2025}
}`}</code>
              </pre>
            </div>
          </div>

          {/* Call to Action */}
          <div className="text-center">
            <div className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20 rounded-xl p-8">
              <h2 className="text-2xl font-bold text-white mb-4">
                Ready to Explore SDLM?
              </h2>
              <p className="text-slate-300 mb-6 leading-relaxed">
                Join the community and contribute to the future of efficient language modeling.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <a
                  href="https://github.com/OpenGVLab/SDLM"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-xl hover:from-purple-700 hover:to-blue-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-purple-500/25"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                  </svg>
                  Star on GitHub
                </a>
                <a
                  href="https://arxiv.org/pdf/2509.24007"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-semibold rounded-xl hover:from-blue-700 hover:to-cyan-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl hover:shadow-blue-500/25"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                  Read Paper
                </a>
                <Link
                  href="/"
                  className="inline-flex items-center gap-2 px-8 py-4 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-purple-500 hover:text-purple-400 transition-all duration-300 transform hover:scale-105"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                  </svg>
                  Back to Home
                </Link>
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
