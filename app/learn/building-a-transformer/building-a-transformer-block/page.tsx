import { LessonPage } from "@/components/lesson-page";

export default function BuildingATransformerBlockPage() {
  return (
    <LessonPage
      contentPath="building-a-transformer/building-a-transformer-block"
      prevLink={{ href: "/learn/building-a-transformer/rope-positional-encoding", label: "← Previous: RoPE Positional Encoding" }}
      nextLink={{ href: "/learn/building-a-transformer/the-final-linear-layer", label: "Next: The Final Linear Layer →" }}
    />
  );
}

