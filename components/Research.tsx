export default function Research() {
  const researchAreas = [
    {
      title: "AI Safety & Alignment",
      description: "Developing techniques to ensure AI systems remain beneficial and aligned with human values as they become more capable.",
      icon: "🛡️"
    },
    {
      title: "Superintelligence Theory",
      description: "Exploring the theoretical foundations of superintelligent systems and their potential capabilities and limitations.",
      icon: "🧠"
    },
    {
      title: "AI Governance",
      description: "Researching policy frameworks and governance structures for responsible AI development and deployment.",
      icon: "⚖️"
    },
    {
      title: "Technical AI Safety",
      description: "Building robust AI systems with interpretability, reliability, and safety guarantees.",
      icon: "🔧"
    },
    {
      title: "AI Ethics",
      description: "Investigating ethical implications of advanced AI systems and developing principled approaches to AI development.",
      icon: "🌍"
    },
    {
      title: "Collaborative AI",
      description: "Creating AI systems that can work effectively with humans and other AI systems in complex environments.",
      icon: "🤝"
    }
  ]

  return (
    <section className="py-20 bg-gray-50">
      <div className="max-w-6xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Research Areas
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Our interdisciplinary research spans multiple domains critical to the development of beneficial superintelligence.
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {researchAreas.map((area, index) => (
            <div key={index} className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
              <div className="text-4xl mb-4">{area.icon}</div>
              <h3 className="text-xl font-semibold mb-3 text-gray-900">{area.title}</h3>
              <p className="text-gray-600">{area.description}</p>
            </div>
          ))}
        </div>
        
        <div className="text-center mt-12">
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition-colors">
            View All Publications
          </button>
        </div>
      </div>
    </section>
  )
}
