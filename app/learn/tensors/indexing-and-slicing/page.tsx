import { LessonPage } from "@/components/lesson-page";

export default function IndexingAndSlicingPage() {
  return (
    <LessonPage
      contentPath="tensors/indexing-and-slicing"
      prevLink={{ href: "/learn/tensors/reshaping-tensors", label: "← Previous: Reshaping Tensors" }}
      nextLink={{ href: "/learn/tensors/concatenating-tensors", label: "Next: Concatenating Tensors →" }}
    />
  );
}

