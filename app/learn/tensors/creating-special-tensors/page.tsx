import { LessonPage } from "@/components/lesson-page";

export default function CreatingSpecialTensorsPage() {
  return (
    <LessonPage
      contentPath="tensors/creating-special-tensors"
      prevLink={{ href: "/learn/tensors/concatenating-tensors", label: "← Previous: Concatenating Tensors" }}
      nextLink={{ href: "/learn/neural-networks/introduction", label: "Next: Neural Networks →" }}
    />
  );
}

