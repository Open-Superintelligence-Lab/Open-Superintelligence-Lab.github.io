import { LessonPage } from "@/components/lesson-page";

export default function FunctionsPage() {
  return (
    <LessonPage
      contentPath="math/functions"
      prevLink={{ href: "/learn/math/derivatives", label: "← Previous: Derivatives" }}
      nextLink={{ href: "/learn/neural-networks/introduction", label: "Next: Neural Networks →" }}
    />
  );
}

