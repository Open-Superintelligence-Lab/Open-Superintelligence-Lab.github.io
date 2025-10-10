import { LessonPage } from "@/components/lesson-page";

export default function FullTransformerInCodePage() {
  return (
    <LessonPage
      contentPath="building-a-transformer/full-transformer-in-code"
      prevLink={{ href: "/learn/building-a-transformer/the-final-linear-layer", label: "← Previous: The Final Linear Layer" }}
      nextLink={{ href: "/learn/building-a-transformer/training-a-transformer", label: "Next: Training a Transformer →" }}
    />
  );
}

