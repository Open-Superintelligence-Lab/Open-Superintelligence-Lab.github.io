import Link from "next/link";

export default function BlogPost() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <header className="container mx-auto px-6 py-8">
        <nav className="flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold hover:text-purple-300 transition-colors">
            Open Superintelligence Lab
          </Link>
          <div className="flex gap-6">
            <Link href="/about" className="hover:text-purple-300 transition-colors">About</Link>
            <Link href="/blog" className="hover:text-purple-300 transition-colors">Blog</Link>
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-purple-300 transition-colors">GitHub</a>
          </div>
        </nav>
      </header>

      {/* Blog Post Content */}
      <main className="container mx-auto px-6 py-16">
        <div className="max-w-4xl mx-auto">
          {/* Back to Blog */}
          <Link href="/blog" className="text-purple-300 hover:text-purple-200 transition-colors mb-8 inline-block">
            ← Back to Blog
          </Link>

          {/* Article Header */}
          <header className="mb-12">
            <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Benign Superintelligence: Open or Closed?
            </h1>
            <div className="flex gap-4 text-gray-400 mb-8">
              <span>January 15, 2024</span>
              <span>•</span>
              <span>8 min read</span>
            </div>
            <p className="text-xl text-gray-300 leading-relaxed">
              Exploring the critical question of whether superintelligence should be developed openly or behind closed doors, 
              and the implications for humanity&apos;s future.
            </p>
          </header>

          {/* Article Content */}
          <article className="prose prose-lg prose-invert max-w-none">
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 border border-white/20 mb-8">
              <h2 className="text-3xl font-bold mb-6 text-purple-300">The Great Debate</h2>
              <p className="text-lg leading-relaxed text-gray-200 mb-6">
                As we stand on the precipice of artificial general intelligence (AGI) and superintelligence, 
                one of the most critical questions facing humanity is whether these transformative technologies 
                should be developed in the open or behind closed doors. This isn&apos;t just a technical decision—it&apos;s 
                a fundamental choice about the future of human civilization.
              </p>
            </div>

            <div className="space-y-8">
              <section>
                <h2 className="text-3xl font-bold mb-4 text-purple-300">The Case for Open Development</h2>
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 mb-6">
                  <h3 className="text-xl font-semibold mb-3 text-purple-200">Democratization of Power</h3>
                  <p className="text-gray-300 leading-relaxed">
                    Open development ensures that superintelligence doesn&apos;t become the exclusive domain of a few 
                    powerful corporations or governments. By making the technology accessible to everyone, we 
                    prevent the concentration of unprecedented power in the hands of a select few.
                  </p>
                </div>
                
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 mb-6">
                  <h3 className="text-xl font-semibold mb-3 text-purple-200">Collective Intelligence</h3>
                  <p className="text-gray-300 leading-relaxed">
                    The wisdom of crowds applies to AI safety as well. With thousands of researchers, ethicists, 
                    and concerned citizens able to review and contribute to the development process, we can 
                    identify potential risks and solutions that might be missed by a small, closed team.
                  </p>
                </div>

                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 mb-6">
                  <h3 className="text-xl font-semibold mb-3 text-purple-200">Transparency and Trust</h3>
                  <p className="text-gray-300 leading-relaxed">
                    Open development builds trust through transparency. When the public can see how AI systems 
                    are being developed and what safety measures are in place, it reduces fear and builds 
                    confidence in the technology.
                  </p>
                </div>
              </section>

              <section>
                <h2 className="text-3xl font-bold mb-4 text-purple-300">The Case for Controlled Development</h2>
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 mb-6">
                  <h3 className="text-xl font-semibold mb-3 text-purple-200">Risk Management</h3>
                  <p className="text-gray-300 leading-relaxed">
                    Superintelligence poses existential risks that could threaten human civilization. Controlled 
                    development allows for careful risk assessment and implementation of safety measures without 
                    the pressure of public scrutiny or competitive pressures.
                  </p>
                </div>

                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 mb-6">
                  <h3 className="text-xl font-semibold mb-3 text-purple-200">Preventing Misuse</h3>
                  <p className="text-gray-300 leading-relaxed">
                    Open source superintelligence could be easily modified for malicious purposes. By keeping 
                    development controlled, we can implement safeguards and prevent the technology from falling 
                    into the wrong hands.
                  </p>
                </div>

                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 mb-6">
                  <h3 className="text-xl font-semibold mb-3 text-purple-200">Quality Control</h3>
                  <p className="text-gray-300 leading-relaxed">
                    Controlled development ensures that only the most qualified researchers work on the technology, 
                    reducing the risk of dangerous mistakes or suboptimal implementations that could have 
                    catastrophic consequences.
                  </p>
                </div>
              </section>

              <section>
                <h2 className="text-3xl font-bold mb-4 text-purple-300">A Middle Path: Open Superintelligence Lab</h2>
                <div className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 backdrop-blur-sm rounded-2xl p-8 border border-purple-500/30 mb-6">
                  <p className="text-lg leading-relaxed text-gray-200 mb-6">
                    At Open Superintelligence Lab, we believe in a third approach: <strong>open development 
                    with built-in safety mechanisms</strong>. Our vision is to democratize superintelligence 
                    development while ensuring that safety and alignment remain paramount.
                  </p>
                  
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-white/10 rounded-xl p-6">
                      <h3 className="text-xl font-semibold mb-3 text-purple-200">Open Source Foundation</h3>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        All our research, code, and findings are open source, allowing global collaboration 
                        and transparency in the development process.
                      </p>
                    </div>
                    <div className="bg-white/10 rounded-xl p-6">
                      <h3 className="text-xl font-semibold mb-3 text-purple-200">Safety-First Architecture</h3>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        Built-in safety mechanisms and alignment protocols ensure that superintelligence 
                        remains beneficial to humanity.
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              <section>
                <h2 className="text-3xl font-bold mb-4 text-purple-300">The Future We&apos;re Building</h2>
                <p className="text-lg leading-relaxed text-gray-200 mb-6">
                  The question of open vs. closed development isn&apos;t just about technology—it&apos;s about the kind 
                  of future we want to create. Do we want a world where superintelligence is controlled by 
                  a few powerful entities, or one where it&apos;s developed collaboratively for the benefit of all?
                </p>
                
                <p className="text-lg leading-relaxed text-gray-200 mb-8">
                  At Open Superintelligence Lab, we&apos;re choosing the latter. We&apos;re building a future where 
                  superintelligence is developed openly, safely, and for the benefit of all humanity. 
                  Join us in creating this future.
                </p>

                <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl p-6 text-center">
                  <h3 className="text-xl font-bold mb-3">Ready to Contribute?</h3>
                  <p className="text-gray-200 mb-4">
                    Help us build the future of open superintelligence
                  </p>
                  <a
                    href="https://github.com/open-superintelligence-lab"
                    className="bg-white text-purple-600 px-6 py-3 rounded-full font-semibold hover:bg-gray-100 transition-colors inline-block"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Join Our Mission
                  </a>
                </div>
              </section>
            </div>
          </article>
        </div>
      </main>

      {/* Footer */}
      <footer className="container mx-auto px-6 py-8 border-t border-white/20">
        <div className="flex justify-between items-center text-gray-400">
          <p>&copy; 2024 Open Superintelligence Lab. Open source for everyone.</p>
          <div className="flex gap-6">
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-white transition-colors">GitHub</a>
            <Link href="/about" className="hover:text-white transition-colors">About</Link>
            <Link href="/blog" className="hover:text-white transition-colors">Blog</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
