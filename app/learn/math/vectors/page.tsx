import { LessonPage } from "@/components/lesson-page";

export default function VectorsPage() {
  return (
    <LessonPage
      contentPath="math/vectors"
      prevLink={{ href: "/learn/math/derivatives", label: "← Previous: Derivatives" }}
      nextLink={{ href: "/learn/math/matrices", label: "Next: Matrices →" }}
    />
  );
}

