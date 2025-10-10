import { LessonPage } from "@/components/lesson-page";

export default function TheDeepseekMlpPage() {
  return (
    <LessonPage
      contentPath="transformer-feedforward/the-deepseek-mlp"
      prevLink={{ href: "/learn/transformer-feedforward/moe-in-code", label: "← Previous: MoE in Code" }}
      nextLink={{ href: "/learn/building-a-transformer/transformer-architecture", label: "Next: Transformer Architecture →" }}
    />
  );
}

