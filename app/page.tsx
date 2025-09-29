
import Link from "next/link";
import { Navigation } from "@/components/navigation";

export default function Home() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation />
      
      <main className="container mx-auto px-6 py-16">
        <div className="text-center max-w-4xl mx-auto">
          <h1 className="text-6xl font-bold mb-6">
            Open Superintelligence Lab
          </h1>
          <h2 className="text-2xl text-gray-400 mb-16">
            开放超级智能实验室
          </h2>
          
          <div className="space-y-8">
            <h3 className="text-3xl font-semibold mb-8">Projects</h3>
            
            <div className="grid gap-6 max-w-2xl mx-auto">
              <Link 
                href="/projects/train-llm-for-1-usd"
                className="block p-8 border border-gray-800 rounded-lg hover:border-gray-600 transition-colors group"
              >
                <h4 className="text-2xl font-semibold mb-4 group-hover:text-gray-300 transition-colors">
                  Train LLM For $1 USD
                </h4>
                <p className="text-gray-400">
                  Revolutionary approach to training large language models at unprecedented cost efficiency
                </p>
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
