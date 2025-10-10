import { LessonPage } from "@/components/lesson-page";

export default function TheActivationFunctionPage() {
  return (
    <LessonPage
      contentPath="neuron-from-scratch/the-activation-function"
      prevLink={{ href: "/learn/neuron-from-scratch/the-linear-step", label: "← Previous: The Linear Step" }}
      nextLink={{ href: "/learn/neuron-from-scratch/building-a-neuron-in-python", label: "Next: Building a Neuron in Python →" }}
    />
  );
}

