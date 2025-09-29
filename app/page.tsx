
'use client';

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
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              {t.description}
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
