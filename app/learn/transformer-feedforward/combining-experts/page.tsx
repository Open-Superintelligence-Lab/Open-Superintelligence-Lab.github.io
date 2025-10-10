import { LessonPage } from "@/components/lesson-page";

export default function CombiningExpertsPage() {
  return (
    <LessonPage
      contentPath="transformer-feedforward/combining-experts"
      prevLink={{ href: "/learn/transformer-feedforward/the-gate", label: "← Previous: The Gate" }}
      nextLink={{ href: "/learn/transformer-feedforward/moe-in-a-transformer", label: "Next: MoE in a Transformer →" }}
    />
  );
}

