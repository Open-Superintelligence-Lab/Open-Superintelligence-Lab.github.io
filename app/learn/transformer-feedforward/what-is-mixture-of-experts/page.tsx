import { LessonPage } from "@/components/lesson-page";

export default function WhatIsMixtureOfExpertsPage() {
  return (
    <LessonPage
      contentPath="transformer-feedforward/what-is-mixture-of-experts"
      prevLink={{ href: "/learn/transformer-feedforward/the-feedforward-layer", label: "← Previous: The Feedforward Layer" }}
      nextLink={{ href: "/learn/transformer-feedforward/the-expert", label: "Next: The Expert →" }}
    />
  );
}

