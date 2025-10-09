import { LessonPage } from "@/components/lesson-page";

export default function IntroductionPage() {
  return (
    <LessonPage
      contentPath="neural-networks/introduction"
      prevLink={{ href: "/learn/math/functions", label: "← Previous: Functions" }}
      nextLink={{ href: "/learn/neural-networks/forward-propagation", label: "Next: Forward Propagation →" }}
    />
  );
}

