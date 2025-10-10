import { LessonPage } from "@/components/lesson-page";

export default function ReshapingTensorsPage() {
  return (
    <LessonPage
      contentPath="tensors/reshaping-tensors"
      prevLink={{ href: "/learn/tensors/transposing-tensors", label: "← Previous: Transposing Tensors" }}
      nextLink={{ href: "/learn/tensors/indexing-and-slicing", label: "Next: Indexing and Slicing →" }}
    />
  );
}

