import Link from "next/link";

export default function About() {
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="container mx-auto px-6 py-8">
        <nav className="flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold hover:text-gray-400 transition-colors">
            Open Superintelligence Lab
          </Link>
          <div className="flex gap-6">
            <Link href="/about" className="text-gray-400">About</Link>
            <Link href="/blog" className="hover:text-gray-400 transition-colors">Blog</Link>
            <a href="https://github.com/Open-Superintelligence-Lab/blueberry-llm" className="hover:text-gray-400 transition-colors" target="_blank" rel="noopener noreferrer">Blueberry LLM</a>
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-gray-400 transition-colors" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-16">
        <div className="max-w-4xl mx-auto">
          {/* Back to Home */}
          <Link href="/" className="text-gray-400 hover:text-white transition-colors mb-8 inline-block">
            ← Back to Home
          </Link>

          {/* Page Header */}
          <header className="mb-12">
            <h1 className="text-5xl font-bold mb-6 text-white">About Us</h1>
            <p className="text-xl text-gray-300 leading-relaxed">
              Building the future of open superintelligence through research, collaboration, and innovation.
            </p>
          </header>

          {/* Mission Section */}
          <div className="bg-gray-900 rounded-lg p-8 mb-12 border border-gray-800">
            <h2 className="text-3xl font-bold mb-6 text-white">Our Mission</h2>
            <p className="text-lg leading-relaxed text-gray-200 mb-6">
              Open Superintelligence Lab is dedicated to making superintelligence development accessible to everyone. 
              We believe that the future of AI should be built openly, safely, and for the benefit of all humanity.
            </p>
            <p className="text-lg leading-relaxed text-gray-200">
              Our mission is to democratize AI development by creating tools and frameworks that allow anyone, 
              regardless of technical expertise, to contribute to superintelligence research and development.
            </p>
          </div>

          {/* Vision Section */}
          <div className="bg-gray-900 rounded-lg p-8 mb-12 border border-gray-800">
            <h2 className="text-3xl font-bold mb-6 text-white">Our Vision</h2>
            <p className="text-lg leading-relaxed text-gray-200 mb-6">
              Any company or person (even with no technical experience) should be able to download our repository 
              and run it on their GPU setup - from 1 GPU to 1 million GPUs. The system will automatically 
              detect your hardware configuration, tune hyperparameters for optimal performance, and run the best 
              possible training with or without manual configuration.
            </p>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-gray-800 rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-3 text-white">🚀 Auto-Scaling</h3>
                <p className="text-gray-300 text-sm leading-relaxed">
                  Seamlessly scale from single GPU to massive distributed clusters
                </p>
              </div>
              <div className="bg-gray-800 rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-3 text-white">⚡ Auto-Tuning</h3>
                <p className="text-gray-300 text-sm leading-relaxed">
                  Intelligent hyperparameter optimization for your hardware
                </p>
              </div>
              <div className="bg-gray-800 rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-3 text-white">🔧 Zero-Config</h3>
                <p className="text-gray-300 text-sm leading-relaxed">
                  Works out of the box with automatic hardware detection
                </p>
              </div>
            </div>
          </div>

          {/* Research Focus */}
          <div className="bg-gray-900 rounded-lg p-8 mb-12 border border-gray-800">
            <h2 className="text-3xl font-bold mb-6 text-white">Research Focus</h2>
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold mb-4 text-white">Blueberry LLM 🫐</h3>
                <p className="text-gray-300 leading-relaxed mb-4">
                  Our flagship Mixture of Experts (MoE) language model implementation. 
                  Clone, install dependencies, and train your own language model with a single command.
                </p>
                <a
                  href="https://github.com/Open-Superintelligence-Lab/blueberry-llm"
                  className="text-gray-400 hover:text-white transition-colors font-semibold"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  → Explore Blueberry LLM
                </a>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-4 text-white">Open Research</h3>
                <p className="text-gray-300 leading-relaxed mb-4">
                  We conduct open research in AI safety, alignment, and governance. 
                  All our findings, code, and models are publicly available.
                </p>
                <a
                  href="https://github.com/Open-Superintelligence-Lab/blueberry-llm/issues"
                  className="text-gray-400 hover:text-white transition-colors font-semibold"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  → Pick a Research Task
                </a>
              </div>
            </div>
          </div>

          {/* Get Involved */}
          <div className="bg-gray-900 rounded-lg p-8 mb-12 border border-gray-800">
            <h2 className="text-3xl font-bold mb-6 text-white">Get Involved</h2>
            <p className="text-lg leading-relaxed text-gray-200 mb-8">
              We believe that addressing the challenges of superintelligence requires a collaborative effort 
              from the global community. Whether you&apos;re a researcher, student, developer, or simply 
              interested in AI safety, there are many ways to contribute.
            </p>
            
            <div className="flex gap-4 justify-center flex-col sm:flex-row">
              <a
                href="https://github.com/Open-Superintelligence-Lab/blueberry-llm/issues"
                className="bg-white text-black px-8 py-4 rounded-lg font-semibold transition-all hover:bg-gray-200"
                target="_blank"
                rel="noopener noreferrer"
              >
                Pick a Research Task
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

          {/* Values */}
          <div className="bg-gray-900 rounded-lg p-8 border border-gray-800">
            <h2 className="text-3xl font-bold mb-6 text-white">Our Values</h2>
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold mb-3 text-white">Open Source</h3>
                <p className="text-gray-300 leading-relaxed">
                  All our research, code, and findings are open source. We believe in transparency 
                  and collaboration as the foundation of safe AI development.
                </p>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-3 text-white">Speed</h3>
                <p className="text-gray-300 leading-relaxed">
                  We optimize for performance and efficiency, ensuring our tools can scale 
                  from single GPU setups to massive distributed systems seamlessly.
                </p>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-3 text-white">Accessibility</h3>
                <p className="text-gray-300 leading-relaxed">
                  We make advanced AI research accessible to everyone, regardless of technical 
                  background or resources.
                </p>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-3 text-white">Innovation</h3>
                <p className="text-gray-300 leading-relaxed">
                  We push the boundaries of what&apos;s possible in AI research while maintaining 
                  our commitment to safety and openness.
                </p>
              </div>
            </div>
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
            <Link href="/about" className="hover:text-white transition-colors">About</Link>
            <Link href="/blog" className="hover:text-white transition-colors">Blog</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
