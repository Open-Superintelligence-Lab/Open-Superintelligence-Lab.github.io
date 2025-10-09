import { LessonPage } from "@/components/lesson-page";

export default function ForwardPropagationPage() {
  return (
    <LessonPage
      contentPath="neural-networks/forward-propagation"
      prevLink={{ href: "/learn/neural-networks/introduction", label: "← Previous: Introduction" }}
      nextLink={{ href: "/learn/neural-networks/backpropagation", label: "Next: Backpropagation →" }}
    />
  );
}

