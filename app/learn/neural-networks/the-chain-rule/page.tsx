import { LessonPage } from "@/components/lesson-page";

export default function TheChainRulePage() {
  return (
    <LessonPage
      contentPath="neural-networks/the-chain-rule"
      prevLink={{ href: "/learn/neural-networks/implementing-a-network", label: "← Previous: Implementing a Network" }}
      nextLink={{ href: "/learn/neural-networks/calculating-gradients", label: "Next: Calculating Gradients →" }}
    />
  );
}

