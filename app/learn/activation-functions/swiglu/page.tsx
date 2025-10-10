import { LessonPage } from "@/components/lesson-page";

export default function SwigluPage() {
  return (
    <LessonPage
      contentPath="activation-functions/swiglu"
      prevLink={{ href: "/learn/activation-functions/silu", label: "← Previous: SiLU" }}
      nextLink={{ href: "/learn/activation-functions/softmax", label: "Next: Softmax →" }}
    />
  );
}

