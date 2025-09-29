import Link from "next/link";
import { Navigation } from "@/components/navigation";

export default function TrainLLMProject() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation currentPath="/projects" />
      
      <main className="container mx-auto px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <Link 
              href="/projects" 
              className="text-gray-400 hover:text-white transition-colors"
            >
              ← Back to Projects
            </Link>
          </div>
          
          <h1 className="text-5xl font-bold mb-6">
            Train LLM For $1 USD
          </h1>
          
          <div className="prose prose-invert max-w-none">
            <p className="text-xl text-gray-300 mb-8">
              Revolutionary approach to training large language models at unprecedented cost efficiency
            </p>
            
            <div className="space-y-6">
              <section>
                <h2 className="text-3xl font-semibold mb-4">Overview</h2>
                <p className="text-gray-300 leading-relaxed">
                  This project explores innovative techniques to dramatically reduce the cost of training 
                  large language models while maintaining or improving performance. By leveraging advanced 
                  optimization algorithms, efficient data processing, and novel architectural approaches, 
                  we aim to make AI training accessible to everyone.
                </p>
              </section>
              
              <section>
                <h2 className="text-3xl font-semibold mb-4">Key Features</h2>
                <ul className="text-gray-300 space-y-2">
                  <li>• Ultra-low cost training methodology</li>
                  <li>• Efficient parameter optimization</li>
                  <li>• Advanced data preprocessing techniques</li>
                  <li>• Novel architectural innovations</li>
                  <li>• Open-source implementation</li>
                </ul>
              </section>
              
              <section>
                <h2 className="text-3xl font-semibold mb-4">Status</h2>
                <p className="text-gray-300">
                  Currently in active development. Follow our progress on GitHub for the latest updates.
                </p>
              </section>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
