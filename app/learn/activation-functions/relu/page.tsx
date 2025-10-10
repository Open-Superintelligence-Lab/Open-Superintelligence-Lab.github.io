import { LessonPage } from "@/components/lesson-page";

export default function ReluPage() {
  return (
    <LessonPage
      contentPath="activation-functions/relu"
      prevLink={{ href: "/learn/neuron-from-scratch/the-concept-of-learning", label: "← Previous: The Concept of Learning" }}
      nextLink={{ href: "/learn/activation-functions/sigmoid", label: "Next: Sigmoid →" }}
    />
  );
}

