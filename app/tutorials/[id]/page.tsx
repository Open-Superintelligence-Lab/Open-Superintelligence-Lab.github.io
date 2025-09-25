import { TutorialViewer } from "@/components/tutorial-viewer";
import { Navigation } from "@/components/navigation";
import { Id } from "@/convex/_generated/dataModel";

interface TutorialPageProps {
  params: {
    id: string;
  };
}

export default function TutorialPage({ params }: TutorialPageProps) {
  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation currentPath="/tutorials" />
      <TutorialViewer tutorialId={params.id as Id<"tutorials">} />
    </div>
  );
}
