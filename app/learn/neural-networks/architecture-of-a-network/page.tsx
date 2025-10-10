import { LessonPage } from "@/components/lesson-page";

export default function ArchitectureOfANetworkPage() {
  return (
    <LessonPage
      contentPath="neural-networks/architecture-of-a-network"
      prevLink={{ href: "/learn/activation-functions/softmax", label: "← Previous: Softmax" }}
      nextLink={{ href: "/learn/neural-networks/building-a-layer", label: "Next: Building a Layer →" }}
    />
  );
}

