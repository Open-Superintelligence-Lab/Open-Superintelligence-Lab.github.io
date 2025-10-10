import { LessonPage } from "@/components/lesson-page";

export default function FunctionsPage() {
  return (
    <LessonPage
      contentPath="math/functions"
      prevLink={{ href: "/learn", label: "← Back to Course" }}
      nextLink={{ href: "/learn/math/derivatives", label: "Next: Derivatives →" }}
    />
  );
}

