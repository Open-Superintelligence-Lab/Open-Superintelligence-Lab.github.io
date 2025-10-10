import { LessonPage } from "@/components/lesson-page";

export default function CalculatingAttentionScoresPage() {
  return (
    <LessonPage
      contentPath="attention-mechanism/calculating-attention-scores"
      prevLink={{ href: "/learn/attention-mechanism/self-attention-from-scratch", label: "← Previous: Self Attention from Scratch" }}
      nextLink={{ href: "/learn/attention-mechanism/applying-attention-weights", label: "Next: Applying Attention Weights →" }}
    />
  );
}

