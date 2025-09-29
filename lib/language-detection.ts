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
    about: 'About',
    github: 'GitHub',
    toggleLanguage: '中文',
    projects: 'Research Projects',
    deepseekTitle: 'DeepSeek-V3.2-Exp Research',
    deepseekDescription: 'Open source research on DeepSeek Sparse Attention (DSA) and long-context efficiency improvements',
    deepseekStatus: 'Active Research',
    gptOssTitle: 'GPT-OSS Research',
    gptOssDescription: 'OpenAI\'s open-source MoE language models with advanced reasoning capabilities and safety features',
    gptOssStatus: 'Open Source',
    learnMore: 'Learn More',
    researchPath: 'Research Path',
    researchQuestions: 'Research Questions',
    contributions: 'How to Contribute',
    openSource: 'Open Source Research',
    deepseekResearchPath: 'Research Path',
    deepseekQuestions: 'Research Questions',
    deepseekContributions: 'How to Contribute',
    deepseekOpenSource: 'Open Source Research',
    gptOssResearchPath: 'Research Path',
    gptOssQuestions: 'Research Questions',
    gptOssContributions: 'How to Contribute',
    gptOssOpenSource: 'Open Source Research'
  },
  zh: {
    title: '开放超级智能实验室',
    subtitle: 'Open Superintelligence Lab',
    description: '通过创新的人工智能方法推进AI研究和开发。',
    about: '关于',
    github: 'GitHub',
    toggleLanguage: 'English',
    projects: '研究项目',
    deepseekTitle: 'DeepSeek-V3.2-Exp 研究',
    deepseekDescription: '关于DeepSeek稀疏注意力机制(DSA)和长上下文效率改进的开源研究',
    deepseekStatus: '活跃研究',
    gptOssTitle: 'GPT-OSS 研究',
    gptOssDescription: 'OpenAI的开源MoE语言模型，具有先进的推理能力和安全特性',
    gptOssStatus: '开源',
    learnMore: '了解更多',
    researchPath: '研究路径',
    researchQuestions: '研究问题',
    contributions: '如何贡献',
    openSource: '开源研究',
    deepseekResearchPath: '研究路径',
    deepseekQuestions: '研究问题',
    deepseekContributions: '如何贡献',
    deepseekOpenSource: '开源研究',
    gptOssResearchPath: '研究路径',
    gptOssQuestions: '研究问题',
    gptOssContributions: '如何贡献',
    gptOssOpenSource: '开源研究'
  }
};
