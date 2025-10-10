import { LessonPage } from "@/components/lesson-page";

export default function TheExpertPage() {
  return (
    <LessonPage
      contentPath="transformer-feedforward/the-expert"
      prevLink={{ href: "/learn/transformer-feedforward/what-is-mixture-of-experts", label: "← Previous: What is Mixture of Experts" }}
      nextLink={{ href: "/learn/transformer-feedforward/the-gate", label: "Next: The Gate →" }}
    />
  );
}

