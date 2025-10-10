import { LessonPage } from "@/components/lesson-page";

export default function SoftmaxPage() {
  return (
    <LessonPage
      contentPath="activation-functions/softmax"
      prevLink={{ href: "/learn/activation-functions/swiglu", label: "← Previous: SwiGLU" }}
      nextLink={{ href: "/learn/neural-networks/introduction", label: "Next: Neural Networks →" }}
    />
  );
}

