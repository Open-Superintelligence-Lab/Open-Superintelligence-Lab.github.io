import { LessonPage } from "@/components/lesson-page";

export default function ApplyingAttentionWeightsPage() {
  return (
    <LessonPage
      contentPath="attention-mechanism/applying-attention-weights"
      prevLink={{ href: "/learn/attention-mechanism/calculating-attention-scores", label: "← Previous: Calculating Attention Scores" }}
      nextLink={{ href: "/learn/attention-mechanism/multi-head-attention", label: "Next: Multi Head Attention →" }}
    />
  );
}

