import { LessonPage } from "@/components/lesson-page";

export default function SelfAttentionFromScratchPage() {
  return (
    <LessonPage
      contentPath="attention-mechanism/self-attention-from-scratch"
      prevLink={{ href: "/learn/attention-mechanism/what-is-attention", label: "← Previous: What is Attention" }}
      nextLink={{ href: "/learn/attention-mechanism/calculating-attention-scores", label: "Next: Calculating Attention Scores →" }}
    />
  );
}

