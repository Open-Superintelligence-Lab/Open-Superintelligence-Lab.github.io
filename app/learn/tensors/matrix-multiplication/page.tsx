import { LessonPage } from "@/components/lesson-page";

export default function MatrixMultiplicationPage() {
  return (
    <LessonPage
      contentPath="tensors/matrix-multiplication"
      prevLink={{ href: "/learn/tensors/tensor-addition", label: "← Previous: Tensor Addition" }}
      nextLink={{ href: "/learn/tensors/transposing-tensors", label: "Next: Transposing Tensors →" }}
    />
  );
}

