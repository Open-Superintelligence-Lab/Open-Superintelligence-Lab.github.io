import { LessonPage } from "@/components/lesson-page";

export default function TrainingPage() {
  return (
    <LessonPage
      contentPath="neural-networks/training"
      prevLink={{ href: "/learn/neural-networks/backpropagation", label: "← Previous: Backpropagation" }}
    />
  );
}
