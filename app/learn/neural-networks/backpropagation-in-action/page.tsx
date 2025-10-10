import { LessonPage } from "@/components/lesson-page";

export default function BackpropagationInActionPage() {
  return (
    <LessonPage
      contentPath="neural-networks/backpropagation-in-action"
      prevLink={{ href: "/learn/neural-networks/calculating-gradients", label: "← Previous: Calculating Gradients" }}
      nextLink={{ href: "/learn/neural-networks/implementing-backpropagation", label: "Next: Implementing Backpropagation →" }}
    />
  );
}

