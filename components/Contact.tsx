export default function Contact() {
  return (
    <section className="py-20 bg-white">
      <div className="max-w-4xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Get Involved
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Join our community of researchers, engineers, and thinkers working towards beneficial superintelligence.
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-12">
          <div>
            <h3 className="text-2xl font-semibold mb-6 text-gray-900">Contact Information</h3>
            <div className="space-y-4">
              <div className="flex items-center">
                <svg className="w-6 h-6 text-blue-600 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <span className="text-gray-600">contact@opensuperintelligence.org</span>
              </div>
              
              <div className="flex items-center">
                <svg className="w-6 h-6 text-blue-600 mr-3" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                <span className="text-gray-600">github.com/Open-Superintelligence-Lab</span>
              </div>
              
              <div className="flex items-center">
                <svg className="w-6 h-6 text-blue-600 mr-3" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/>
                </svg>
                <span className="text-gray-600">@OpenSuperAI</span>
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="text-2xl font-semibold mb-6 text-gray-900">Join Our Community</h3>
            <div className="space-y-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">Research Collaborators</h4>
                <p className="text-gray-600 text-sm">
                  Join our research team and contribute to cutting-edge AI safety research.
                </p>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">Open Source Contributors</h4>
                <p className="text-gray-600 text-sm">
                  Help us build tools and frameworks for safe AI development.
                </p>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">Community Members</h4>
                <p className="text-gray-600 text-sm">
                  Participate in discussions, workshops, and educational programs.
                </p>
              </div>
            </div>
            
            <button className="mt-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition-colors w-full">
              Join Our Discord
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}
