import { LessonPage } from "@/components/lesson-page";

export default function MoeInCodePage() {
  return (
    <LessonPage
      contentPath="transformer-feedforward/moe-in-code"
      prevLink={{ href: "/learn/transformer-feedforward/moe-in-a-transformer", label: "← Previous: MoE in a Transformer" }}
      nextLink={{ href: "/learn/transformer-feedforward/the-deepseek-mlp", label: "Next: The DeepSeek MLP →" }}
    />
  );
}

