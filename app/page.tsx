
'use client';

import Link from "next/link";
import { Navigation } from "@/components/navigation";
import { useLanguage } from "@/components/providers/language-provider";
import { translations } from "@/lib/language-detection";

export default function Home() {
  const { language } = useLanguage();
  const t = translations[language];

  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation />
      
      <main className="container mx-auto px-6 py-16">
        <div className="text-center max-w-4xl mx-auto">
          <h1 className="text-6xl font-bold mb-6">
            {t.title}
          </h1>
          <h2 className="text-2xl text-gray-400 mb-16">
            {t.subtitle}
          </h2>
          
          <div className="space-y-8">
            <p className="text-xl text-gray-300 max-w-2xl mx-auto mb-16">
              {t.description}
            </p>
            
            <div className="space-y-8">
              <h3 className="text-3xl font-semibold mb-8">{t.projects}</h3>
              
              <div className="grid gap-6 max-w-3xl mx-auto">
                <Link 
                  href="/research/deepseek-v3-2-exp"
                  className="block p-8 border border-gray-800 rounded-lg hover:border-gray-600 transition-colors group"
                >
                  <h4 className="text-2xl font-semibold mb-4 group-hover:text-gray-300 transition-colors">
                    {t.deepseekTitle}
                  </h4>
                  <p className="text-gray-400 mb-4">
                    {t.deepseekDescription}
                  </p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center text-sm text-gray-500">
                      <span className="bg-green-500/20 text-green-400 px-2 py-1 rounded mr-3">{t.deepseekStatus}</span>
                      <span>Open Source Research</span>
                    </div>
                    <span className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors">
                      {t.learnMore} →
                    </span>
                  </div>
                </Link>

                <Link 
                  href="/research/gpt-oss"
                  className="block p-8 border border-gray-800 rounded-lg hover:border-gray-600 transition-colors group"
                >
                  <h4 className="text-2xl font-semibold mb-4 group-hover:text-gray-300 transition-colors">
                    {t.gptOssTitle}
                  </h4>
                  <p className="text-gray-400 mb-4">
                    {t.gptOssDescription}
                  </p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center text-sm text-gray-500">
                      <span className="bg-blue-500/20 text-blue-400 px-2 py-1 rounded mr-3">{t.gptOssStatus}</span>
                      <span>OpenAI Research</span>
                    </div>
                    <span className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors">
                      {t.learnMore} →
                    </span>
                  </div>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
