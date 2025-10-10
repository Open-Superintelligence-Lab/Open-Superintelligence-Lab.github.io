import { LessonPage } from "@/components/lesson-page";

export default function MatricesPage() {
  return (
    <LessonPage
      contentPath="math/matrices"
      prevLink={{ href: "/learn/math/vectors", label: "← Previous: Vectors" }}
      nextLink={{ href: "/learn/math/gradients", label: "Next: Gradients →" }}
    />
  );
}

