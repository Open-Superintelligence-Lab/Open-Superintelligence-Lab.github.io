import { LessonPage } from "@/components/lesson-page";

export default function ConcatenatingTensorsPage() {
  return (
    <LessonPage
      contentPath="tensors/concatenating-tensors"
      prevLink={{ href: "/learn/tensors/indexing-and-slicing", label: "← Previous: Indexing and Slicing" }}
      nextLink={{ href: "/learn/tensors/creating-special-tensors", label: "Next: Creating Special Tensors →" }}
    />
  );
}

