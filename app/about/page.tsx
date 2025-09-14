export default function About() {
  return (
    <div className="font-sans min-h-screen p-8 pb-20 sm:p-20">
      <main className="max-w-4xl mx-auto">
        <div className="mb-6">
          <a 
            href="/"
            className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 underline"
          >
            ← Back to Home
          </a>
        </div>
        <h1 className="text-4xl font-bold mb-8 text-center">About Us</h1>
        
        <div className="space-y-6 text-lg leading-relaxed">
          <p>
            Welcome to the Open Superintelligence Lab! We are a research organization 
            dedicated to advancing the field of artificial intelligence and ensuring 
            the safe development of superintelligent systems.
          </p>
          
          <p>
            Our mission is to conduct cutting-edge research in AI safety, alignment, 
            and governance while fostering collaboration between researchers, 
            policymakers, and industry leaders.
          </p>
          
          <h2 className="text-2xl font-semibold mt-8 mb-4">Our Focus Areas</h2>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>AI Safety and Alignment Research</li>
            <li>Superintelligence Governance</li>
            <li>Technical AI Safety</li>
            <li>Policy and Ethics</li>
            <li>Public Education and Outreach</li>
          </ul>
          
          <h2 className="text-2xl font-semibold mt-8 mb-4">Get Involved</h2>
          <p>
            We believe that addressing the challenges of superintelligence requires 
            a collaborative effort from the global community. Whether you're a 
            researcher, student, policymaker, or simply interested in AI safety, 
            there are many ways to contribute to our mission.
          </p>
          
          <div className="mt-8 p-6 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <p className="text-center text-gray-600 dark:text-gray-400">
              This is a test About page created to demonstrate the Next.js routing system.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
