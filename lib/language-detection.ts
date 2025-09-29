// lib/language-detection.ts
export type Language = 'en' | 'zh';

export interface LanguageContext {
  language: Language;
  setLanguage: (lang: Language) => void;
}

// Simple IP-based language detection
export async function detectLanguageFromIP(): Promise<Language> {
  try {
    // Use a free IP geolocation service
    const response = await fetch('https://ipapi.co/json/');
    const data = await response.json();
    
    // Check if user is from Hong Kong or other Chinese-speaking regions
    const chineseRegions = ['HK', 'CN', 'TW', 'MO', 'SG'];
    return chineseRegions.includes(data.country_code) ? 'zh' : 'en';
  } catch (error) {
    console.log('Failed to detect language from IP, defaulting to English');
    return 'en';
  }
}

// Translations
export const translations = {
  en: {
    title: 'Open Superintelligence Lab',
    subtitle: '开放超级智能实验室',
    description: 'Advancing AI research and development through innovative approaches to artificial intelligence.',
    skool: 'Skool',
    github: 'GitHub',
    toggleLanguage: '中文'
  },
  zh: {
    title: '开放超级智能实验室',
    subtitle: 'Open Superintelligence Lab',
    description: '通过创新的人工智能方法推进AI研究和开发。',
    skool: 'Skool',
    github: 'GitHub',
    toggleLanguage: 'English'
  }
};
