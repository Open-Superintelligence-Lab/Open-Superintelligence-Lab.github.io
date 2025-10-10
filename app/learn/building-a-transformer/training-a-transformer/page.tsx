import { LessonPage } from "@/components/lesson-page";

export default function TrainingATransformerPage() {
  return (
    <LessonPage
      contentPath="building-a-transformer/training-a-transformer"
      prevLink={{ href: "/learn/building-a-transformer/full-transformer-in-code", label: "← Previous: Full Transformer in Code" }}
      nextLink={{ href: "/learn", label: "Next: Course Complete →" }}
    />
  );
}

