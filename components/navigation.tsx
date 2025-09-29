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
    <header className="container mx-auto px-6 py-8">
      <nav className="flex justify-between items-center">
        <Link href="/" className="text-2xl font-bold hover:text-gray-400 transition-colors">
          {t.title}
        </Link>
        <div className="flex gap-6 items-center">
          <button
            onClick={toggleLanguage}
            className="text-sm px-3 py-1 border border-gray-600 rounded hover:border-gray-400 hover:text-gray-300 transition-colors"
          >
            {t.toggleLanguage}
          </button>
          <a 
            href="https://skool.com/open-superintelligence-lab" 
            className="hover:text-gray-400 transition-colors" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            {t.skool}
          </a>
          <a 
            href="https://github.com/open-superintelligence-lab" 
            className="hover:text-gray-400 transition-colors" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            {t.github}
          </a>
        </div>
      </nav>
    </header>
  );
}

