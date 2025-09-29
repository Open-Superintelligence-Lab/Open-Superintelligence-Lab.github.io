import Link from "next/link";
import { Navigation } from "@/components/navigation";

export default function ProjectsPage() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation currentPath="/projects" />
      
      <main className="container mx-auto px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <Link 
              href="/" 
              className="text-gray-400 hover:text-white transition-colors"
            >
              ← Back to Home
            </Link>
          </div>
          
          <h1 className="text-5xl font-bold mb-6">Projects</h1>
          <p className="text-xl text-gray-400 mb-12">
            Our research initiatives advancing the field of artificial intelligence
          </p>
          
          <div className="grid gap-8 max-w-3xl mx-auto">
            <Link 
              href="/projects/train-llm-for-1-usd"
              className="block p-8 border border-gray-800 rounded-lg hover:border-gray-600 transition-colors group"
            >
              <h2 className="text-3xl font-semibold mb-4 group-hover:text-gray-300 transition-colors">
                Train LLM For $1 USD
              </h2>
              <p className="text-gray-400 mb-4">
                Revolutionary approach to training large language models at unprecedented cost efficiency
              </p>
              <div className="flex items-center text-sm text-gray-500">
                <span className="bg-green-500/20 text-green-400 px-2 py-1 rounded mr-3">Active</span>
                <span>In Development</span>
              </div>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
