import { LessonPage } from "@/components/lesson-page";

export default function MultiHeadAttentionPage() {
  return (
    <LessonPage
      contentPath="attention-mechanism/multi-head-attention"
      prevLink={{ href: "/learn/attention-mechanism/applying-attention-weights", label: "← Previous: Applying Attention Weights" }}
      nextLink={{ href: "/learn/attention-mechanism/attention-in-code", label: "Next: Attention in Code →" }}
    />
  );
}

