import { LessonPage } from "@/components/lesson-page";

export default function TransformerArchitecturePage() {
  return (
    <LessonPage
      contentPath="building-a-transformer/transformer-architecture"
      prevLink={{ href: "/learn/transformer-feedforward/the-deepseek-mlp", label: "← Previous: The DeepSeek MLP" }}
      nextLink={{ href: "/learn/building-a-transformer/rope-positional-encoding", label: "Next: RoPE Positional Encoding →" }}
    />
  );
}

