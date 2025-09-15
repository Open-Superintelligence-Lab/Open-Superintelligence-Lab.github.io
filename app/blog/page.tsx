import Link from "next/link";

export default function Blog() {
  const posts = [
    {
      id: "moving-fast",
      title: "Moving Fast: Pick a Task, Make an Impact",
      excerpt: "The best way to contribute to open superintelligence research? Pick a task from our Blueberry LLM issues and start building. Here's why moving fast matters.",
      date: "2024-01-15",
      readTime: "5 min read"
    },
    {
      id: "benign-superintelligence-open-or-closed",
      title: "Benign Superintelligence: Open or Closed?",
      excerpt: "Exploring the critical question of whether superintelligence should be developed openly or behind closed doors, and the implications for humanity's future.",
      date: "2024-01-15",
      readTime: "8 min read"
    }
  ];

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="container mx-auto px-6 py-8">
        <nav className="flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold hover:text-gray-400 transition-colors">
            Open Superintelligence Lab
          </Link>
          <div className="flex gap-6">
            <Link href="/about" className="hover:text-gray-400 transition-colors">About</Link>
            <Link href="/blog" className="text-gray-400">Blog</Link>
            <a href="https://github.com/Open-Superintelligence-Lab/blueberry-llm" className="hover:text-gray-400 transition-colors" target="_blank" rel="noopener noreferrer">Blueberry LLM</a>
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-gray-400 transition-colors" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
        </nav>
      </header>

      {/* Blog Content */}
      <main className="container mx-auto px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl font-bold mb-8 text-white">
            Blog
          </h1>
          <p className="text-xl text-gray-300 mb-12">
            Insights, research, and thoughts on the future of superintelligence
          </p>

          {/* Blog Posts */}
          <div className="space-y-8">
            {posts.map((post) => (
              <article key={post.id} className="bg-gray-900 rounded-lg p-8 border border-gray-800 hover:bg-gray-800 transition-all">
                <Link href={`/blog/${post.id}`}>
                  <h2 className="text-2xl font-bold mb-4 hover:text-gray-400 transition-colors">
                    {post.title}
                  </h2>
                  <p className="text-gray-300 mb-4 leading-relaxed">
                    {post.excerpt}
                  </p>
                  <div className="flex gap-4 text-sm text-gray-400">
                    <span>{post.date}</span>
                    <span>•</span>
                    <span>{post.readTime}</span>
                  </div>
                </Link>
              </article>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="container mx-auto px-6 py-8 border-t border-gray-800">
        <div className="flex justify-between items-center text-gray-400">
          <p>&copy; 2024 Open Superintelligence Lab. Open source for everyone.</p>
          <div className="flex gap-6">
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-white transition-colors" target="_blank" rel="noopener noreferrer">GitHub</a>
            <Link href="/about" className="hover:text-white transition-colors">About</Link>
            <Link href="/blog" className="hover:text-white transition-colors">Blog</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
