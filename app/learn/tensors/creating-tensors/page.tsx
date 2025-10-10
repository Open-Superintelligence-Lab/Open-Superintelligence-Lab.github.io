import { LessonPage } from "@/components/lesson-page";

export default function CreatingTensorsPage() {
  return (
    <LessonPage
      contentPath="tensors/creating-tensors"
      prevLink={{ href: "/learn/math/gradients", label: "← Previous: Gradients" }}
      nextLink={{ href: "/learn/tensors/tensor-addition", label: "Next: Tensor Addition →" }}
    />
  );
}

