import { LessonPage } from "@/components/lesson-page";

export default function SigmoidPage() {
  return (
    <LessonPage
      contentPath="activation-functions/sigmoid"
      prevLink={{ href: "/learn/activation-functions/relu", label: "← Previous: ReLU" }}
      nextLink={{ href: "/learn/activation-functions/tanh", label: "Next: Tanh →" }}
    />
  );
}

