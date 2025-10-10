import { LessonPage } from "@/components/lesson-page";

export default function TheGatePage() {
  return (
    <LessonPage
      contentPath="transformer-feedforward/the-gate"
      prevLink={{ href: "/learn/transformer-feedforward/the-expert", label: "← Previous: The Expert" }}
      nextLink={{ href: "/learn/transformer-feedforward/combining-experts", label: "Next: Combining Experts →" }}
    />
  );
}

