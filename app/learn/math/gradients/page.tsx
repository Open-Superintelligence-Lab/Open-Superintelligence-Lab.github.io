import { LessonPage } from "@/components/lesson-page";

export default function GradientsPage() {
  return (
    <LessonPage
      contentPath="math/gradients"
      prevLink={{ href: "/learn/math/matrices", label: "← Previous: Matrices" }}
      nextLink={{ href: "/learn/tensors/creating-tensors", label: "Next: Creating Tensors →" }}
    />
  );
}

