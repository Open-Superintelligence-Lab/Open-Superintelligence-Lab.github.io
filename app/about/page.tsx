export default function AboutPage() {
  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-6 py-20">
        <h1 className="text-5xl font-bold text-gray-900 mb-8">
          About Open Superintelligence Lab
        </h1>
        
        <div className="prose prose-lg max-w-none">
          <p className="text-xl text-gray-600 mb-6">
            The Open Superintelligence Lab is a research organization dedicated to advancing artificial intelligence 
            towards beneficial superintelligence through open collaboration and responsible innovation.
          </p>
          
          <h2 className="text-3xl font-semibold text-gray-900 mt-12 mb-6">Our Mission</h2>
          <p className="text-gray-600 mb-6">
            We believe that superintelligent AI systems have the potential to solve humanity's greatest challenges, 
            from climate change to disease eradication. However, we also recognize the profound risks associated 
            with such powerful systems. Our mission is to ensure that superintelligence, when achieved, will be 
            beneficial, safe, and aligned with human values.
          </p>
          
          <h2 className="text-3xl font-semibold text-gray-900 mt-12 mb-6">What We Do</h2>
          <ul className="list-disc pl-6 text-gray-600 space-y-2">
            <li>Conduct cutting-edge research in AI safety and alignment</li>
            <li>Develop frameworks for safe superintelligence development</li>
            <li>Build tools and methodologies for AI governance</li>
            <li>Foster international collaboration on AI safety</li>
            <li>Educate the public about superintelligence risks and benefits</li>
          </ul>
          
          <div className="mt-12">
            <a 
              href="/" 
              className="inline-flex items-center bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
            >
              ← Back to Home
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
