import { LessonPage } from "@/components/lesson-page";

export default function TensorAdditionPage() {
  return (
    <LessonPage
      contentPath="tensors/tensor-addition"
      prevLink={{ href: "/learn/tensors/creating-tensors", label: "← Previous: Creating Tensors" }}
      nextLink={{ href: "/learn/tensors/matrix-multiplication", label: "Next: Matrix Multiplication →" }}
    />
  );
}

