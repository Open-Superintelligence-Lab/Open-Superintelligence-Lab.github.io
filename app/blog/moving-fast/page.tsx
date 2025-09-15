import Link from "next/link";

export default function MovingFastBlog() {
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="container mx-auto px-6 py-8">
        <nav className="flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold hover:text-gray-400 transition-colors">
            Open Superintelligence Lab
          </Link>
          <div className="flex gap-6">
            <Link href="/about" className="hover:text-gray-400 transition-colors">About</Link>
            <Link href="/blog" className="hover:text-gray-400 transition-colors">Blog</Link>
            <a href="https://github.com/Open-Superintelligence-Lab/blueberry-llm" className="hover:text-gray-400 transition-colors" target="_blank" rel="noopener noreferrer">Blueberry LLM</a>
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-gray-400 transition-colors" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
        </nav>
      </header>

      {/* Blog Post Content */}
      <main className="container mx-auto px-6 py-16">
        <div className="max-w-4xl mx-auto">
          {/* Back to Blog */}
          <Link href="/blog" className="text-gray-400 hover:text-white transition-colors mb-8 inline-block">
            ← Back to Blog
          </Link>

          {/* Article Header */}
          <header className="mb-12">
            <h1 className="text-5xl font-bold mb-6 text-white">
              Moving Fast: Pick a Task, Make an Impact
            </h1>
            <div className="flex gap-4 text-gray-400 mb-8">
              <span>January 15, 2024</span>
              <span>•</span>
              <span>5 min read</span>
            </div>
            <p className="text-xl text-gray-300 leading-relaxed">
              The best way to contribute to open superintelligence research? Pick a task from our 
              <a href="https://github.com/Open-Superintelligence-Lab/blueberry-llm/issues" className="text-gray-400 hover:text-white transition-colors"> Blueberry LLM issues</a> 
              and start building. Here&apos;s why moving fast matters and how you can get involved today.
            </p>
          </header>

          {/* Article Content */}
          <article className="prose prose-lg prose-invert max-w-none">
            <div className="bg-gray-900 rounded-lg p-8 border border-gray-800 mb-8">
              <h2 className="text-3xl font-bold mb-6 text-white">Why Speed Matters in AI Research</h2>
              <p className="text-lg leading-relaxed text-gray-200 mb-6">
                In the race toward superintelligence, every day counts. While others debate theoretical frameworks, 
                we&apos;re building. While others plan perfect architectures, we&apos;re iterating. The future belongs 
                to those who ship, not those who speculate.
              </p>
            </div>

            <div className="space-y-8">
              <section>
                <h2 className="text-3xl font-bold mb-4 text-white">Ready-to-Go Research Tasks</h2>
                <div className="bg-gray-900 rounded-lg p-6 border border-gray-800 mb-6">
                  <h3 className="text-xl font-semibold mb-3 text-white">🔬 Research: More Small vs Few Big Experts</h3>
                  <p className="text-gray-300 leading-relaxed mb-4">
                    Draw scaling laws comparing architectures with many small experts versus fewer large experts. 
                    This is perfect for newcomers and will generate valuable insights for the field.
                  </p>
                  <a
                    href="https://github.com/Open-Superintelligence-Lab/blueberry-llm/issues/1"
                    className="text-gray-400 hover:text-white transition-colors font-semibold"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    → Take this task
                  </a>
                </div>

                <div className="bg-gray-900 rounded-lg p-6 border border-gray-800 mb-6">
                  <h3 className="text-xl font-semibold mb-3 text-white">⚡ Research: Activation Function Ablation</h3>
                  <p className="text-gray-300 leading-relaxed mb-4">
                    Test SwiGLU, GEGLU, SiLU, GELU, ReLU2 and other activation functions in our MoE architecture. 
                    Another great first issue for newcomers to contribute meaningful research.
                  </p>
                  <a
                    href="https://github.com/Open-Superintelligence-Lab/blueberry-llm/issues/2"
                    className="text-gray-400 hover:text-white transition-colors font-semibold"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    → Take this task
                  </a>
                </div>

                <div className="bg-gray-900 rounded-lg p-6 border border-gray-800 mb-6">
                  <h3 className="text-xl font-semibold mb-3 text-white">📊 Research: Batch Scheduling & Curriculum</h3>
                  <p className="text-gray-300 leading-relaxed mb-4">
                    Implement length bucketing and perplexity curriculum learning. This advanced research task 
                    will help optimize training efficiency and model performance.
                  </p>
                  <a
                    href="https://github.com/Open-Superintelligence-Lab/blueberry-llm/issues/3"
                    className="text-gray-400 hover:text-white transition-colors font-semibold"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    → Take this task
                  </a>
                </div>
              </section>

              <section>
                <h2 className="text-3xl font-bold mb-4 text-white">How to Get Started</h2>
                <div className="bg-gray-900 rounded-lg p-8 border border-gray-800 mb-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-gray-800 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-3 text-white">1. Fork & Clone</h3>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        Fork the <a href="https://github.com/Open-Superintelligence-Lab/blueberry-llm" className="text-gray-400 hover:text-white transition-colors">Blueberry LLM repository</a> and clone it locally.
                      </p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-3 text-white">2. Pick Your Task</h3>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        Browse our <a href="https://github.com/Open-Superintelligence-Lab/blueberry-llm/issues" className="text-gray-400 hover:text-white transition-colors">open issues</a> and pick one that matches your skills.
                      </p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-3 text-white">3. Build & Experiment</h3>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        Run experiments, test hypotheses, and push the boundaries of what&apos;s possible.
                      </p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-3 text-white">4. Submit PR</h3>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        Share your findings with the community and help advance superintelligence research.
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              <section>
                <h2 className="text-3xl font-bold mb-4 text-white">The Philosophy of Fast Iteration</h2>
                <p className="text-lg leading-relaxed text-gray-200 mb-6">
                  We believe in the power of rapid experimentation. Every failed experiment teaches us something. 
                  Every successful iteration brings us closer to superintelligence. The key is to start building 
                  today, not tomorrow.
                </p>
                
                <p className="text-lg leading-relaxed text-gray-200 mb-8">
                  Don&apos;t wait for the perfect plan. Don&apos;t wait for more resources. Don&apos;t wait for 
                  someone else to solve the problem. Pick a task, start coding, and make an impact.
                </p>

                <div className="bg-gray-900 rounded-lg p-6 text-center border border-gray-800">
                  <h3 className="text-xl font-bold mb-3">Ready to Move Fast?</h3>
                  <p className="text-gray-200 mb-4">
                    Pick a research task and start building the future today
                  </p>
                  <a
                    href="https://github.com/Open-Superintelligence-Lab/blueberry-llm/issues"
                    className="bg-white text-black px-6 py-3 rounded-lg font-semibold hover:bg-gray-200 transition-colors inline-block"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Browse Open Tasks
                  </a>
                </div>
              </section>
            </div>
          </article>
        </div>
      </main>

      {/* Footer */}
      <footer className="container mx-auto px-6 py-8 border-t border-gray-800">
        <div className="flex justify-between items-center text-gray-400">
          <p>&copy; 2024 Open Superintelligence Lab. Open source for everyone.</p>
          <div className="flex gap-6">
            <a href="https://github.com/Open-Superintelligence-Lab/blueberry-llm" className="hover:text-white transition-colors" target="_blank" rel="noopener noreferrer">Blueberry LLM</a>
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-white transition-colors" target="_blank" rel="noopener noreferrer">GitHub</a>
            <Link href="/about" className="hover:text-white transition-colors">About</Link>
            <Link href="/blog" className="hover:text-white transition-colors">Blog</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
