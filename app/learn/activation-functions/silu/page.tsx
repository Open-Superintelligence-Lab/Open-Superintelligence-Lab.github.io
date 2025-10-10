import { LessonPage } from "@/components/lesson-page";

export default function SiluPage() {
  return (
    <LessonPage
      contentPath="activation-functions/silu"
      prevLink={{ href: "/learn/activation-functions/tanh", label: "← Previous: Tanh" }}
      nextLink={{ href: "/learn/activation-functions/swiglu", label: "Next: SwiGLU →" }}
    />
  );
}

