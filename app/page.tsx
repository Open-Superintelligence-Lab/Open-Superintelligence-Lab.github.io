import Image from "next/image";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <header className="container mx-auto px-6 py-8">
        <nav className="flex justify-between items-center">
          <div className="text-2xl font-bold">Open Superintelligence Lab</div>
          <div className="flex gap-6">
            <a href="/about" className="hover:text-purple-300 transition-colors">About</a>
            <a href="/blog" className="hover:text-purple-300 transition-colors">Blog</a>
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-purple-300 transition-colors">GitHub</a>
          </div>
        </nav>
      </header>

      {/* Hero Section */}
      <main className="container mx-auto px-6 py-16">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-6xl font-bold mb-8 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            Open Superintelligence Lab
          </h1>
          <p className="text-xl text-gray-300 mb-12 leading-relaxed">
            Democratizing AI development with accessible, scalable superintelligence research
          </p>
          
          {/* Vision Section */}
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 mb-12 border border-white/20">
            <h2 className="text-3xl font-bold mb-6 text-purple-300">Our Vision</h2>
            <p className="text-lg leading-relaxed text-gray-200">
              Any company or person (even with no technical experience) should be able to download this repository 
              and run it on their GPU setup - from 1 GPU to 1 million GPUs. The system will be able to automatically 
              detect your hardware configuration, tune hyperparameters for optimal performance, and run the best 
              possible training with or without manual configuration from your side.
            </p>
          </div>

          {/* Key Features */}
          <div className="grid md:grid-cols-3 gap-8 mb-12">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <div className="text-4xl mb-4">🚀</div>
              <h3 className="text-xl font-semibold mb-3">Auto-Scaling</h3>
              <p className="text-gray-300">Seamlessly scale from single GPU to massive distributed clusters</p>
            </div>
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <div className="text-4xl mb-4">⚡</div>
              <h3 className="text-xl font-semibold mb-3">Auto-Tuning</h3>
              <p className="text-gray-300">Intelligent hyperparameter optimization for your hardware</p>
            </div>
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <div className="text-4xl mb-4">🔧</div>
              <h3 className="text-xl font-semibold mb-3">Zero-Config</h3>
              <p className="text-gray-300">Works out of the box with automatic hardware detection</p>
            </div>
          </div>

          {/* Call to Action */}
          <div className="flex gap-4 justify-center flex-col sm:flex-row">
            <a
              href="https://github.com/open-superintelligence-lab"
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 px-8 py-4 rounded-full font-semibold transition-all transform hover:scale-105"
              target="_blank"
              rel="noopener noreferrer"
            >
              Get Started on GitHub
            </a>
            <a
              href="/blog"
              className="border border-white/30 hover:bg-white/10 px-8 py-4 rounded-full font-semibold transition-all"
            >
              Read Our Blog
            </a>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="container mx-auto px-6 py-8 border-t border-white/20">
        <div className="flex justify-between items-center text-gray-400">
          <p>&copy; 2024 Open Superintelligence Lab. Open source for everyone.</p>
          <div className="flex gap-6">
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-white transition-colors">GitHub</a>
            <a href="/about" className="hover:text-white transition-colors">About</a>
            <a href="/blog" className="hover:text-white transition-colors">Blog</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
