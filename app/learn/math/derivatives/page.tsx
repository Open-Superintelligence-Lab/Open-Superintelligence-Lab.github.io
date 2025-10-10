import { LessonPage } from "@/components/lesson-page";

export default function DerivativesPage() {
  return (
    <LessonPage
      contentPath="math/derivatives"
      prevLink={{ href: "/learn/math/functions", label: "← Previous: Functions" }}
      nextLink={{ href: "/learn/math/vectors", label: "Next: Vectors →" }}
    />
  );
}

