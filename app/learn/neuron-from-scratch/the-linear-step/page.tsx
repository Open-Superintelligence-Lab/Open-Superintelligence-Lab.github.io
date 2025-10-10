import { LessonPage } from "@/components/lesson-page";

export default function TheLinearStepPage() {
  return (
    <LessonPage
      contentPath="neuron-from-scratch/the-linear-step"
      prevLink={{ href: "/learn/neuron-from-scratch/what-is-a-neuron", label: "← Previous: What is a Neuron" }}
      nextLink={{ href: "/learn/neuron-from-scratch/the-activation-function", label: "Next: The Activation Function →" }}
    />
  );
}

