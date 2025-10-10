import { LessonPage } from "@/components/lesson-page";

export default function BuildingANeuronInPythonPage() {
  return (
    <LessonPage
      contentPath="neuron-from-scratch/building-a-neuron-in-python"
      prevLink={{ href: "/learn/neuron-from-scratch/the-activation-function", label: "← Previous: The Activation Function" }}
      nextLink={{ href: "/learn/neuron-from-scratch/making-a-prediction", label: "Next: Making a Prediction →" }}
    />
  );
}

