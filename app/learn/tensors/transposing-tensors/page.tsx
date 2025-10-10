import { LessonPage } from "@/components/lesson-page";

export default function TransposingTensorsPage() {
  return (
    <LessonPage
      contentPath="tensors/transposing-tensors"
      prevLink={{ href: "/learn/tensors/matrix-multiplication", label: "← Previous: Matrix Multiplication" }}
      nextLink={{ href: "/learn/tensors/reshaping-tensors", label: "Next: Reshaping Tensors →" }}
    />
  );
}

