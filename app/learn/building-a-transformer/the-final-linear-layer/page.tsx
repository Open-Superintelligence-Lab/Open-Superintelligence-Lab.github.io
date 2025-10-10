import { LessonPage } from "@/components/lesson-page";

export default function TheFinalLinearLayerPage() {
  return (
    <LessonPage
      contentPath="building-a-transformer/the-final-linear-layer"
      prevLink={{ href: "/learn/building-a-transformer/building-a-transformer-block", label: "← Previous: Building a Transformer Block" }}
      nextLink={{ href: "/learn/building-a-transformer/full-transformer-in-code", label: "Next: Full Transformer in Code →" }}
    />
  );
}

