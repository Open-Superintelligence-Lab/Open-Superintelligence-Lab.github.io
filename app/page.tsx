
export default function Home() {
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="container mx-auto px-6 py-8">
        <nav className="flex justify-between items-center">
          <div className="text-2xl font-bold">Open Superintelligence Lab</div>
          <div className="flex gap-6">
            <a href="/about" className="hover:text-gray-400 transition-colors">About</a>
            <a href="/blog" className="hover:text-gray-400 transition-colors">Blog</a>
            <a href="https://github.com/Open-Superintelligence-Lab/blueberry-llm" className="hover:text-gray-400 transition-colors" target="_blank" rel="noopener noreferrer">Blueberry LLM</a>
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-gray-400 transition-colors" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
        </nav>
      </header>

      {/* Hero Section */}
      <main className="container mx-auto px-6 py-16">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-6xl font-bold mb-8 text-white">
            Open Superintelligence Lab
          </h1>
          <p className="text-xl text-gray-300 mb-12 leading-relaxed">
            Making AI development accessible with scalable superintelligence research
          </p>
          
          {/* Vision Section */}
          <div className="bg-gray-900 rounded-lg p-8 mb-12 border border-gray-800">
            <h2 className="text-3xl font-bold mb-6 text-white">Our Vision</h2>
            <p className="text-lg leading-relaxed text-gray-200">
              Any company or person (even with no technical experience) should be able to download this repository 
              and run it on their GPU setup - from 1 GPU to 1 million GPUs. The system will be able to automatically 
              detect your hardware configuration, tune hyperparameters for optimal performance, and run the best 
              possible training with or without manual configuration from your side.
            </p>
          </div>

          {/* Key Features */}
          <div className="grid md:grid-cols-3 gap-8 mb-12">
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="text-4xl mb-4">🚀</div>
              <h3 className="text-xl font-semibold mb-3">Auto-Scaling</h3>
              <p className="text-gray-300">Seamlessly scale from single GPU to massive distributed clusters</p>
            </div>
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="text-4xl mb-4">⚡</div>
              <h3 className="text-xl font-semibold mb-3">Auto-Tuning</h3>
              <p className="text-gray-300">Intelligent hyperparameter optimization for your hardware</p>
            </div>
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="text-4xl mb-4">🔧</div>
              <h3 className="text-xl font-semibold mb-3">Zero-Config</h3>
              <p className="text-gray-300">Works out of the box with automatic hardware detection</p>
            </div>
          </div>

          {/* Blueberry LLM Section */}
          <div className="bg-gray-900 rounded-lg p-8 mb-12 border border-gray-800">
            <h2 className="text-3xl font-bold mb-6 text-white">Blueberry LLM 🫐</h2>
            <p className="text-lg leading-relaxed text-gray-200 mb-6">
              Our flagship Mixture of Experts (MoE) language model implementation. Clone, install dependencies, 
              and train your own language model with a single command. Perfect for researchers and developers 
              looking to experiment with cutting-edge AI architectures.
            </p>
            <div className="flex gap-4 justify-center flex-col sm:flex-row">
              <a
                href="https://github.com/Open-Superintelligence-Lab/blueberry-llm"
                className="bg-white text-black px-8 py-4 rounded-lg font-semibold transition-all hover:bg-gray-200"
                target="_blank"
                rel="noopener noreferrer"
              >
                Blueberry LLM GitHub
              </a>
              <a
                href="https://github.com/Open-Superintelligence-Lab/blueberry-llm/issues"
                className="border border-gray-600 hover:bg-gray-800 px-8 py-4 rounded-lg font-semibold transition-all"
                target="_blank"
                rel="noopener noreferrer"
              >
                Pick a Research Task
              </a>
            </div>
          </div>

          {/* Call to Action */}
          <div className="flex gap-4 justify-center flex-col sm:flex-row">
            <a
              href="/blog"
              className="bg-white text-black px-8 py-4 rounded-lg font-semibold transition-all hover:bg-gray-200"
            >
              Read Our Blog
            </a>
            <a
              href="https://github.com/open-superintelligence-lab"
              className="border border-gray-600 hover:bg-gray-800 px-8 py-4 rounded-lg font-semibold transition-all"
              target="_blank"
              rel="noopener noreferrer"
            >
              Join Our Community
            </a>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="container mx-auto px-6 py-8 border-t border-gray-800">
        <div className="flex justify-between items-center text-gray-400">
          <p>&copy; 2024 Open Superintelligence Lab. Open source for everyone.</p>
          <div className="flex gap-6">
            <a href="https://github.com/Open-Superintelligence-Lab/blueberry-llm" className="hover:text-white transition-colors" target="_blank" rel="noopener noreferrer">Blueberry LLM</a>
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-white transition-colors" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a href="/about" className="hover:text-white transition-colors">About</a>
            <a href="/blog" className="hover:text-white transition-colors">Blog</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
