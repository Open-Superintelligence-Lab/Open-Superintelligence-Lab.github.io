'use client';

import Link from "next/link";

interface NavigationProps {
  currentPath?: string;
}

export function Navigation({ }: NavigationProps) {
  return (
    <header className="absolute top-0 left-0 right-0 z-50">
      <div className="container mx-auto px-6 py-6">
        <nav className="flex justify-between items-center">
          <Link href="/" className="flex items-center gap-3 text-2xl font-bold hover:text-blue-400 transition-colors">
            <div className="relative">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                <span className="text-white text-lg">🔮</span>
              </div>
              {/* Subtle glow effect */}
              <div className="absolute inset-0 w-10 h-10 bg-gradient-to-r from-blue-500/10 to-blue-600/10 rounded-xl blur-sm"></div>
            </div>
          </Link>
          <div className="flex gap-2 items-center">
            <Link
              href="/"
              className="px-3 py-2 text-sm hover:text-purple-400 transition-colors"
            >
              Home
            </Link>
            <Link
              href="/learn"
              className="px-3 py-2 text-sm hover:text-purple-400 transition-colors"
            >
              Learn
            </Link>
            <a
              href="https://discord.com/invite/6AbXGpKTwN"
              className="px-3 py-2 text-sm hover:text-blue-400 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              Discord
            </a>
            <a
              href="https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g"
              className="px-3 py-2 text-sm hover:text-red-500 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              YouTube
            </a>
          </div>
        </nav>
      </div>
    </header>
  );
}

