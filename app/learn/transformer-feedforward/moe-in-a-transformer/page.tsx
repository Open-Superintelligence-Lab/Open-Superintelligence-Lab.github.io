import { LessonPage } from "@/components/lesson-page";

export default function MoeInATransformerPage() {
  return (
    <LessonPage
      contentPath="transformer-feedforward/moe-in-a-transformer"
      prevLink={{ href: "/learn/transformer-feedforward/combining-experts", label: "← Previous: Combining Experts" }}
      nextLink={{ href: "/learn/transformer-feedforward/moe-in-code", label: "Next: MoE in Code →" }}
    />
  );
}

