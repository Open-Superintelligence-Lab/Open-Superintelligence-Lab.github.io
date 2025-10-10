import { LessonPage } from "@/components/lesson-page";

export default function WhatIsAttentionPage() {
  return (
    <LessonPage
      contentPath="attention-mechanism/what-is-attention"
      prevLink={{ href: "/learn/neural-networks/implementing-backpropagation", label: "← Previous: Implementing Backpropagation" }}
      nextLink={{ href: "/learn/attention-mechanism/self-attention-from-scratch", label: "Next: Self Attention from Scratch →" }}
    />
  );
}

