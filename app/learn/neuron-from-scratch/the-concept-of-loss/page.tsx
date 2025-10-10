import { LessonPage } from "@/components/lesson-page";

export default function TheConceptOfLossPage() {
  return (
    <LessonPage
      contentPath="neuron-from-scratch/the-concept-of-loss"
      prevLink={{ href: "/learn/neuron-from-scratch/making-a-prediction", label: "← Previous: Making a Prediction" }}
      nextLink={{ href: "/learn/neuron-from-scratch/the-concept-of-learning", label: "Next: The Concept of Learning →" }}
    />
  );
}

