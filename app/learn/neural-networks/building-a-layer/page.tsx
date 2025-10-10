import { LessonPage } from "@/components/lesson-page";

export default function BuildingALayerPage() {
  return (
    <LessonPage
      contentPath="neural-networks/building-a-layer"
      prevLink={{ href: "/learn/neural-networks/architecture-of-a-network", label: "← Previous: Architecture of a Network" }}
      nextLink={{ href: "/learn/neural-networks/implementing-a-network", label: "Next: Implementing a Network →" }}
    />
  );
}

