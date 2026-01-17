'use client';

import Link from "next/link";

interface NavigationProps {
  currentPath?: string;
}

export function Navigation({ }: NavigationProps) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50">
      {/* Glassmorphism Navigation Bar */}
      <div className="border-b border-white/10 bg-slate-900/80 backdrop-blur-xl">
        <div className="container mx-auto px-6 py-4">
          <nav className="flex justify-between items-center">
            {/* Logo */}
            <Link href="/" className="group flex items-center gap-3 hover:scale-105 transition-all duration-300">
              <div className="relative">
                {/* Animated gradient border */}
                <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 via-purple-600 to-cyan-600 rounded-2xl opacity-75 group-hover:opacity-100 blur group-hover:blur-md transition-all duration-300 animate-pulse"></div>

                {/* Logo container */}
                <div className="relative w-11 h-11 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center shadow-2xl">
                  <span className="text-2xl filter drop-shadow-lg">🔮</span>
                </div>
              </div>

              {/* Brand Name */}
              <div className="hidden md:flex flex-col">
                <span className="text-sm font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                  Open Superintelligence
                </span>
                <span className="text-xs text-slate-400">Building the Future</span>
              </div>
            </Link>

            {/* Navigation Links */}
            <div className="flex gap-1 items-center">
              <Link
                href="/"
                className="group relative px-4 py-2 text-sm font-medium text-slate-300 hover:text-white transition-all duration-300"
              >
                <span className="relative z-10">Home</span>
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600/0 via-purple-600/0 to-cyan-600/0 group-hover:from-blue-600/10 group-hover:via-purple-600/10 group-hover:to-cyan-600/10 rounded-lg transition-all duration-300"></div>
              </Link>

              <Link
                href="/learn"
                className="group relative px-4 py-2 text-sm font-medium text-slate-300 hover:text-white transition-all duration-300"
              >
                <span className="relative z-10">Learn</span>
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600/0 via-purple-600/0 to-cyan-600/0 group-hover:from-blue-600/10 group-hover:via-purple-600/10 group-hover:to-cyan-600/10 rounded-lg transition-all duration-300"></div>
              </Link>

              {/* Social Links with Icons */}
              <a
                href="https://discord.com/invite/6AbXGpKTwN"
                className="group relative px-4 py-2 text-sm font-medium text-slate-300 hover:text-blue-400 transition-all duration-300"
                target="_blank"
                rel="noopener noreferrer"
              >
                <span className="relative z-10 flex items-center gap-2">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515a.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0a12.64 12.64 0 0 0-.617-1.25a.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057a19.9 19.9 0 0 0 5.993 3.03a.078.078 0 0 0 .084-.028a14.09 14.09 0 0 0 1.226-1.994a.076.076 0 0 0-.041-.106a13.107 13.107 0 0 1-1.872-.892a.077.077 0 0 1-.008-.128a10.2 10.2 0 0 0 .372-.292a.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127a12.299 12.299 0 0 1-1.873.892a.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028a19.839 19.839 0 0 0 6.002-3.03a.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.956-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.955-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.946 2.418-2.157 2.418z" />
                  </svg>
                  <span className="hidden sm:inline">Discord</span>
                </span>
                <div className="absolute inset-0 bg-blue-600/0 group-hover:bg-blue-600/10 rounded-lg transition-all duration-300"></div>
              </a>

              <a
                href="https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g"
                className="group relative px-4 py-2 text-sm font-medium text-slate-300 hover:text-red-500 transition-all duration-300"
                target="_blank"
                rel="noopener noreferrer"
              >
                <span className="relative z-10 flex items-center gap-2">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
                  </svg>
                  <span className="hidden sm:inline">YouTube</span>
                </span>
                <div className="absolute inset-0 bg-red-600/0 group-hover:bg-red-600/10 rounded-lg transition-all duration-300"></div>
              </a>
            </div>
          </nav>
        </div>
      </div>
    </header>
  );
}

