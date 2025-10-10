import { LessonPage } from "@/components/lesson-page";

export default function TheConceptOfLearningPage() {
  return (
    <LessonPage
      contentPath="neuron-from-scratch/the-concept-of-learning"
      prevLink={{ href: "/learn/neuron-from-scratch/the-concept-of-loss", label: "← Previous: The Concept of Loss" }}
      nextLink={{ href: "/learn/neural-networks/introduction", label: "Next: Neural Networks →" }}
    />
  );
}

