import { LessonPage } from "@/components/lesson-page";

export default function WhatIsANeuronPage() {
  return (
    <LessonPage
      contentPath="neuron-from-scratch/what-is-a-neuron"
      prevLink={{ href: "/learn/tensors/creating-special-tensors", label: "← Previous: Creating Special Tensors" }}
      nextLink={{ href: "/learn/neuron-from-scratch/the-linear-step", label: "Next: The Linear Step →" }}
    />
  );
}

