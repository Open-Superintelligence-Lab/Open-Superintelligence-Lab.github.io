import { LessonPage } from "@/components/lesson-page";

export default function BackpropagationPage() {
  return (
    <LessonPage
      contentPath="neural-networks/backpropagation"
      prevLink={{ href: "/learn/neural-networks/forward-propagation", label: "← Previous: Forward Propagation" }}
      nextLink={{ href: "/learn/neural-networks/training", label: "Next: Training & Optimization →" }}
    />
  );
}
