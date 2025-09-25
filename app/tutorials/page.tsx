import { TutorialBrowser } from "@/components/tutorial-browser";
import { Navigation } from "@/components/navigation";

export default function TutorialsPage() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation currentPath="/tutorials" />
      <TutorialBrowser />
    </div>
  );
}
