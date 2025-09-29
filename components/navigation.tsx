'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";
import { translations } from "@/lib/language-detection";

interface NavigationProps {
  currentPath?: string;
}

export function Navigation({ currentPath }: NavigationProps) {
  const { language, setLanguage } = useLanguage();
  const t = translations[language];

  const toggleLanguage = () => {
    setLanguage(language === 'en' ? 'zh' : 'en');
  };

  return (
    <header className="container mx-auto px-6 py-6">
      <nav className="flex justify-between items-center">
        <Link href="/" className="flex items-center gap-3 text-2xl font-bold hover:text-blue-400 transition-colors">
          <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">OSI</span>
          </div>
          {t.title}
        </Link>
        <div className="flex gap-4 items-center">
          <button
            onClick={toggleLanguage}
            className="text-sm px-4 py-2 bg-slate-800/50 border border-slate-600/50 rounded-lg hover:border-blue-500/50 hover:bg-slate-700/50 transition-all duration-200"
          >
            {t.toggleLanguage}
          </button>
          <a 
            href="https://skool.com/open-superintelligence-lab" 
            className="px-4 py-2 text-sm hover:text-blue-400 transition-colors" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            {t.skool}
          </a>
          <a 
            href="https://github.com/open-superintelligence-lab" 
            className="px-4 py-2 text-sm hover:text-blue-400 transition-colors" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            {t.github}
          </a>
          <Link 
            href="/learn" 
            className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-sm font-medium rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200"
          >
            Learn
          </Link>
        </div>
      </nav>
    </header>
  );
}

