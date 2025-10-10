import { LessonPage } from "@/components/lesson-page";

export default function TanhPage() {
  return (
    <LessonPage
      contentPath="activation-functions/tanh"
      prevLink={{ href: "/learn/activation-functions/sigmoid", label: "← Previous: Sigmoid" }}
      nextLink={{ href: "/learn/activation-functions/silu", label: "Next: SiLU →" }}
    />
  );
}

