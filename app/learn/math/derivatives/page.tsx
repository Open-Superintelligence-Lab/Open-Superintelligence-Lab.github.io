import { LessonPage } from "@/components/lesson-page";

export default function DerivativesPage() {
  return (
    <LessonPage
      contentPath="math/derivatives"
      prevLink={{ href: "/learn", label: "← Back to Course" }}
      nextLink={{ href: "/learn/math/functions", label: "Next: Functions →" }}
    />
  );
}

