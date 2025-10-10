import { LessonPage } from "@/components/lesson-page";

export default function TheFeedforwardLayerPage() {
  return (
    <LessonPage
      contentPath="transformer-feedforward/the-feedforward-layer"
      prevLink={{ href: "/learn/attention-mechanism/attention-in-code", label: "← Previous: Attention in Code" }}
      nextLink={{ href: "/learn/transformer-feedforward/what-is-mixture-of-experts", label: "Next: What is Mixture of Experts →" }}
    />
  );
}

