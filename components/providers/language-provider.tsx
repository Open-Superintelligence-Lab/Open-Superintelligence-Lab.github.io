'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { Language, LanguageContext, detectLanguageFromIP } from '@/lib/language-detection';

const LanguageContextProvider = createContext<LanguageContext | undefined>(undefined);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguage] = useState<Language>('en');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check localStorage first
    const savedLanguage = localStorage.getItem('language') as Language;
    if (savedLanguage && (savedLanguage === 'en' || savedLanguage === 'zh')) {
      setLanguage(savedLanguage);
      setIsLoading(false);
    } else {
      // Detect from IP if no saved preference
      detectLanguageFromIP().then((detectedLang) => {
        setLanguage(detectedLang);
        localStorage.setItem('language', detectedLang);
        setIsLoading(false);
      });
    }
  }, []);

  const handleSetLanguage = (lang: Language) => {
    setLanguage(lang);
    localStorage.setItem('language', lang);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <LanguageContextProvider.Provider value={{ language, setLanguage: handleSetLanguage }}>
      {children}
    </LanguageContextProvider.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContextProvider);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}
