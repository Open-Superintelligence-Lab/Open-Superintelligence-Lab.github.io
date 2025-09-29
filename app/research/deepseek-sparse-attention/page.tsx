'use client';

import { useLanguage } from "@/components/providers/language-provider";

export default function DeepSeekSparseAttentionPage() {
  const { language } = useLanguage();

  const tasks = language === 'en' ? [
    { text: "Review and understand DeepGEMM PR #200", url: "https://github.com/deepseek-ai/DeepGEMM/pull/200" },
    { text: "Review and understand FlashMLA PR #98", url: "https://github.com/deepseek-ai/FlashMLA/pull/98" },
    { text: "Create markdown explanations and practice exercises for both" },
    { text: "Create YouTube videos explaining both" },
    { text: "Create paid Skool bonus exercises" },
    { text: "Translate all materials to Chinese" },
    { text: "Research repository", url: "https://github.com/Open-Superintelligence-Lab/deepseek-sparse-attention-research" }
  ] : [
    { text: "审查和理解DeepGEMM PR #200", url: "https://github.com/deepseek-ai/DeepGEMM/pull/200" },
    { text: "审查和理解FlashMLA PR #98", url: "https://github.com/deepseek-ai/FlashMLA/pull/98" },
    { text: "为两者创建markdown解释和实践练习" },
    { text: "创建YouTube视频解释两者" },
    { text: "创建付费Skool奖励练习" },
    { text: "将所有材料翻译成中文" },
    { text: "研究仓库", url: "https://github.com/Open-Superintelligence-Lab/deepseek-sparse-attention-research" }
  ];

  return (
    <main className="container mx-auto px-4 sm:px-6 py-8 sm:py-12">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-blue-400">
          {language === 'en' ? 'Tasks' : '任务'}
        </h1>
        <div className="space-y-3">
          {tasks.map((task, index) => (
            <div key={index} className="flex items-start gap-3 text-gray-300">
              <div className="w-2 h-2 bg-blue-400 rounded-full mt-2"></div>
              <span className="leading-relaxed">
                {task.url ? (
                  <a 
                    href={task.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:text-blue-300 underline"
                  >
                    {task.text}
                  </a>
                ) : (
                  task.text
                )}
              </span>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
