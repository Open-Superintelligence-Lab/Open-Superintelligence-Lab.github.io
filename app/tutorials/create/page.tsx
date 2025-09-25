import { TutorialEditor } from "@/components/tutorial-editor";
import { Navigation } from "@/components/navigation";

export default function CreateTutorialPage() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation currentPath="/tutorials/create" />
      <TutorialEditor />
    </div>
  );
}
