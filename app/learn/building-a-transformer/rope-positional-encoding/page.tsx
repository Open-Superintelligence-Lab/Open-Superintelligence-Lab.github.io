import { LessonPage } from "@/components/lesson-page";

export default function RopePositionalEncodingPage() {
  return (
    <LessonPage
      contentPath="building-a-transformer/rope-positional-encoding"
      prevLink={{ href: "/learn/building-a-transformer/transformer-architecture", label: "← Previous: Transformer Architecture" }}
      nextLink={{ href: "/learn/building-a-transformer/building-a-transformer-block", label: "Next: Building a Transformer Block →" }}
    />
  );
}

