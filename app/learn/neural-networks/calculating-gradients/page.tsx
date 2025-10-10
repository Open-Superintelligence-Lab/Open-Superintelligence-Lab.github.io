import { LessonPage } from "@/components/lesson-page";

export default function CalculatingGradientsPage() {
  return (
    <LessonPage
      contentPath="neural-networks/calculating-gradients"
      prevLink={{ href: "/learn/neural-networks/the-chain-rule", label: "← Previous: The Chain Rule" }}
      nextLink={{ href: "/learn/neural-networks/backpropagation-in-action", label: "Next: Backpropagation in Action →" }}
    />
  );
}

