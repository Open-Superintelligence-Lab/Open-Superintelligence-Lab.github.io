export default function Hero() {
  return (
    <section className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900 text-white">
      <div className="text-center max-w-4xl mx-auto px-6">
        <h1 className="text-6xl md:text-8xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
          Open Superintelligence Lab
        </h1>
        <p className="text-xl md:text-2xl mb-8 text-gray-300 max-w-3xl mx-auto">
          Advancing AI research and development towards beneficial superintelligence. 
          Building the future of artificial intelligence through open collaboration and responsible innovation.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition-colors">
            Join Our Research
          </button>
          <button className="border border-white text-white hover:bg-white hover:text-gray-900 font-semibold py-3 px-8 rounded-lg transition-colors">
            Learn More
          </button>
        </div>
      </div>
    </section>
  )
}
