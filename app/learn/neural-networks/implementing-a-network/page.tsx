import { LessonPage } from "@/components/lesson-page";

export default function ImplementingANetworkPage() {
  return (
    <LessonPage
      contentPath="neural-networks/implementing-a-network"
      prevLink={{ href: "/learn/neural-networks/building-a-layer", label: "← Previous: Building a Layer" }}
      nextLink={{ href: "/learn/neural-networks/the-chain-rule", label: "Next: The Chain Rule →" }}
    />
  );
}

