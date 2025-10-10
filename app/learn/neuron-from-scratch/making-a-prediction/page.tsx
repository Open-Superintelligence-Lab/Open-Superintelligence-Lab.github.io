import { LessonPage } from "@/components/lesson-page";

export default function MakingAPredictionPage() {
  return (
    <LessonPage
      contentPath="neuron-from-scratch/making-a-prediction"
      prevLink={{ href: "/learn/neuron-from-scratch/building-a-neuron-in-python", label: "← Previous: Building a Neuron in Python" }}
      nextLink={{ href: "/learn/neuron-from-scratch/the-concept-of-loss", label: "Next: The Concept of Loss →" }}
    />
  );
}

