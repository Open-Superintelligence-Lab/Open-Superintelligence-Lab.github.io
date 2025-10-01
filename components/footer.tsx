'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";
import { translations } from "@/lib/language-detection";

export function Footer() {
  const { language } = useLanguage();
  const t = translations[language];

  return (
    <footer className="bg-slate-900/50 border-t border-slate-700/50 backdrop-blur-sm">
      <div className="container mx-auto px-6 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4">
          {/* Logo and title */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
                <span className="text-white text-sm">🔮</span>
              </div>
              <div className="absolute inset-0 w-8 h-8 bg-gradient-to-r from-blue-500/10 to-blue-600/10 rounded-lg blur-sm"></div>
            </div>
            <div className="text-sm text-slate-300">
              <div className="font-semibold text-white">Open Superintelligence Lab</div>
              <div className="text-xs text-slate-400">开放超级智能实验室</div>
            </div>
          </div>

          {/* Navigation links */}
          <div className="flex gap-6 items-center">
            <Link 
              href="/about" 
              className="text-sm text-slate-300 hover:text-blue-400 transition-colors"
            >
              {t.about}
            </Link>
            <a 
              href="https://github.com/open-superintelligence-lab" 
              className="text-sm text-slate-300 hover:text-blue-400 transition-colors" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              {t.github}
            </a>
            <Link 
              href="/contribute" 
              className="text-sm text-slate-300 hover:text-blue-400 transition-colors"
            >
              {language === 'en' ? 'Contribute' : '贡献'}
            </Link>
            <Link 
              href="/learn" 
              className="text-sm text-slate-300 hover:text-blue-400 transition-colors"
            >
              Learn
            </Link>
          </div>
        </div>

        {/* Bottom section */}
        <div className="mt-6 pt-6 border-t border-slate-700/50 flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-slate-400">
          <div>
            © 2024 Open Superintelligence Lab. {language === 'en' ? 'All rights reserved.' : '保留所有权利。'}
          </div>
          <div className="flex gap-4">
            <a 
              href="https://discord.com/invite/6AbXGpKTwN" 
              className="hover:text-blue-400 transition-colors" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              Discord
            </a>
            <span className="text-slate-600">•</span>
            <span>
              {language === 'en' ? 'Advancing AI research and development' : '推进AI研究和开发'}
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
