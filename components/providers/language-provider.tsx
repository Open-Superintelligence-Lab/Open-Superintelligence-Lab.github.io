'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { Language, LanguageContext } from '@/lib/language-detection';

const LanguageContextProvider = createContext<LanguageContext | undefined>(undefined);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguage] = useState<Language>('en');

  useEffect(() => {
    // Check localStorage for saved preference, default to English
    const savedLanguage = localStorage.getItem('language') as Language;
    if (savedLanguage && (savedLanguage === 'en' || savedLanguage === 'zh' || savedLanguage === 'sr')) {
      setLanguage(savedLanguage);
    }
  }, []);

  const handleSetLanguage = (lang: Language) => {
    setLanguage(lang);
    localStorage.setItem('language', lang);
  };

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
