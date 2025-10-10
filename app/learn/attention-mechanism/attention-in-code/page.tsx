import { LessonPage } from "@/components/lesson-page";

export default function AttentionInCodePage() {
  return (
    <LessonPage
      contentPath="attention-mechanism/attention-in-code"
      prevLink={{ href: "/learn/attention-mechanism/multi-head-attention", label: "← Previous: Multi Head Attention" }}
      nextLink={{ href: "/learn/transformer-feedforward/the-feedforward-layer", label: "Next: The Feedforward Layer →" }}
    />
  );
}

