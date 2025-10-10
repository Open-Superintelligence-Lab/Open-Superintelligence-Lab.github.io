import { LessonPage } from "@/components/lesson-page";

export default function ImplementingBackpropagationPage() {
  return (
    <LessonPage
      contentPath="neural-networks/implementing-backpropagation"
      prevLink={{ href: "/learn/neural-networks/backpropagation-in-action", label: "← Previous: Backpropagation in Action" }}
      nextLink={{ href: "/learn", label: "Next: Course Complete →" }}
    />
  );
}

