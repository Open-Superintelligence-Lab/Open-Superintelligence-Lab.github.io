import Link from "next/link";

export default function Blog() {
  const posts = [
    {
      id: "benign-superintelligence-open-or-closed",
      title: "Benign Superintelligence: Open or Closed?",
      excerpt: "Exploring the critical question of whether superintelligence should be developed openly or behind closed doors, and the implications for humanity's future.",
      date: "2024-01-15",
      readTime: "8 min read"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <header className="container mx-auto px-6 py-8">
        <nav className="flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold hover:text-purple-300 transition-colors">
            Open Superintelligence Lab
          </Link>
          <div className="flex gap-6">
            <Link href="/about" className="hover:text-purple-300 transition-colors">About</Link>
            <Link href="/blog" className="text-purple-300">Blog</Link>
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-purple-300 transition-colors">GitHub</a>
          </div>
        </nav>
      </header>

      {/* Blog Content */}
      <main className="container mx-auto px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl font-bold mb-8 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            Blog
          </h1>
          <p className="text-xl text-gray-300 mb-12">
            Insights, research, and thoughts on the future of superintelligence
          </p>

          {/* Blog Posts */}
          <div className="space-y-8">
            {posts.map((post) => (
              <article key={post.id} className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 border border-white/20 hover:bg-white/15 transition-all">
                <Link href={`/blog/${post.id}`}>
                  <h2 className="text-2xl font-bold mb-4 hover:text-purple-300 transition-colors">
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
      <footer className="container mx-auto px-6 py-8 border-t border-white/20">
        <div className="flex justify-between items-center text-gray-400">
          <p>&copy; 2024 Open Superintelligence Lab. Open source for everyone.</p>
          <div className="flex gap-6">
            <a href="https://github.com/open-superintelligence-lab" className="hover:text-white transition-colors">GitHub</a>
            <Link href="/about" className="hover:text-white transition-colors">About</Link>
            <Link href="/blog" className="hover:text-white transition-colors">Blog</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
