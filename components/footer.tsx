'use client';

import Link from "next/link";

export function Footer() {
  return (
    <footer className="relative bg-[#1f1e1d] border-t border-[#f0eee6]/5 backdrop-blur-xl">
      {/* Subtle overlay */}
      <div className="absolute inset-0 bg-[#f0eee6]/[0.02]"></div>

      <div className="relative container mx-auto px-6 py-12">
        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-8">
          {/* Brand Section */}
          <div className="space-y-4 pt-4">
            <p className="text-sm text-[#f0eee6]/40 leading-relaxed uppercase tracking-widest font-bold">
              Open Superintelligence Lab
            </p>
            <p className="text-sm text-[#f0eee6]/60 leading-relaxed">
              Advancing AI research and development through open collaboration.
            </p>
          </div>

          {/* Quick Links */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Quick Links</h3>
            <nav className="flex flex-col gap-3">
              <Link href="/" className="group flex items-center gap-2 text-sm text-slate-400 hover:text-blue-400 transition-colors">
                <span className="w-1 h-1 bg-slate-600 group-hover:bg-blue-400 rounded-full transition-colors"></span>
                Home
              </Link>
              <Link href="/about" className="group flex items-center gap-2 text-sm text-slate-400 hover:text-blue-400 transition-colors">
                <span className="w-1 h-1 bg-slate-600 group-hover:bg-blue-400 rounded-full transition-colors"></span>
                About
              </Link>
              <Link href="/research" className="group flex items-center gap-2 text-sm text-slate-400 hover:text-blue-400 transition-colors">
                <span className="w-1 h-1 bg-slate-600 group-hover:bg-blue-400 rounded-full transition-colors"></span>
                Research
              </Link>
              <Link href="/learn" className="group flex items-center gap-2 text-sm text-slate-400 hover:text-blue-400 transition-colors">
                <span className="w-1 h-1 bg-slate-600 group-hover:bg-blue-400 rounded-full transition-colors"></span>
                Learn
              </Link>
            </nav>
          </div>

          {/* Community & Social */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Connect</h3>
            <div className="flex flex-col gap-3">
              <a
                href="https://discord.com/invite/6AbXGpKTwN"
                className="group flex items-center gap-3 text-sm text-slate-400 hover:text-blue-400 transition-all duration-300"
                target="_blank"
                rel="noopener noreferrer"
              >
                <div className="p-2 bg-slate-800/50 group-hover:bg-blue-600/20 rounded-lg transition-all duration-300">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515a.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0a12.64 12.64 0 0 0-.617-1.25a.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057a19.9 19.9 0 0 0 5.993 3.03a.078.078 0 0 0 .084-.028a14.09 14.09 0 0 0 1.226-1.994a.076.076 0 0 0-.041-.106a13.107 13.107 0 0 1-1.872-.892a.077.077 0 0 1-.008-.128a10.2 10.2 0 0 0 .372-.292a.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127a12.299 12.299 0 0 1-1.873.892a.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028a19.839 19.839 0 0 0 6.002-3.03a.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.956-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.955-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.946 2.418-2.157 2.418z" />
                  </svg>
                </div>
                <span>Join our Discord</span>
              </a>

              <a
                href="https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g"
                className="group flex items-center gap-3 text-sm text-slate-400 hover:text-red-500 transition-all duration-300"
                target="_blank"
                rel="noopener noreferrer"
              >
                <div className="p-2 bg-slate-800/50 group-hover:bg-red-600/20 rounded-lg transition-all duration-300">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
                  </svg>
                </div>
                <span>Watch on YouTube</span>
              </a>

              <a
                href="https://github.com/open-superintelligence-lab"
                className="group flex items-center gap-3 text-sm text-slate-400 hover:text-purple-400 transition-all duration-300"
                target="_blank"
                rel="noopener noreferrer"
              >
                <div className="p-2 bg-slate-800/50 group-hover:bg-purple-600/20 rounded-lg transition-all duration-300">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                  </svg>
                </div>
                <span>Star on GitHub</span>
              </a>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-[#f0eee6]/5 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-[#f0eee6]/40">
            © 2025 <span className="text-[#f0eee6]/60 font-medium">Open Superintelligence Lab</span>. All rights reserved.
          </p>
          <div className="flex items-center gap-3 text-sm text-slate-500">
            <span className="flex items-center gap-2">
              <svg className="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Budapest, Hungary
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
