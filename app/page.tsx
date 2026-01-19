import fs from "fs";
import path from "path";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getAllPosts, BlogPost } from "@/lib/blog-utils";
import Link from "next/link";

export default function Home() {
  const filePath = path.join(process.cwd(), "ABOUT_LAB.md");
  const aboutContent = fs.readFileSync(filePath, "utf8");
  const posts = getAllPosts();

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden min-h-[30vh] flex flex-col pt-24">
        {/* Updated background to #1f1e1d */}
        <div className="absolute inset-0 bg-[#1f1e1d]"></div>

        {/* Animated gradient mesh overlay - subtle */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse delay-700"></div>
        </div>

        {/* Grid pattern overlay */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(240,238,230,.02)_1px,transparent_1px),linear-gradient(90deg,rgba(240,238,230,.02)_1px,transparent_1px)] bg-[size:72px_72px]"></div>

        <div className="relative container mx-auto px-6 pb-8 flex items-center justify-center">
          <div className="text-center max-w-6xl">
            {/* Main Heading with cream-to-white gradient */}
            <div className="relative mb-4">
              <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight leading-tight">
                <span className="inline-block py-2 bg-gradient-to-b from-[#f0eee6] via-[#f0eee6] to-[#f0eee6]/60 bg-clip-text text-transparent">
                  Open Superintelligence Lab
                </span>
              </h1>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content Rendered from Markdown */}
      <main className="relative bg-[#1f1e1d] pb-32">
        <div className="container mx-auto px-6 max-w-3xl">
          <div className="markdown-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({ node, ...props }: any) => <h1 className="text-4xl font-bold text-[#f0eee6] mb-8 border-b border-[#f0eee6]/10 pb-4" {...props} />,
                h2: ({ node, ...props }: any) => <h2 className="text-2xl font-bold text-[#f0eee6] mt-12 mb-6" {...props} />,
                h3: ({ node, ...props }: any) => <h3 className="text-xl font-bold text-[#f0eee6] mt-8 mb-4 transition-colors" {...props} />,
                p: ({ node, ...props }: any) => <p className="text-[#f0eee6]/70 text-lg leading-relaxed mb-6" {...props} />,
                ul: ({ node, ...props }: any) => <ul className="list-disc list-inside mb-6 space-y-3 text-[#f0eee6]/70 text-lg" {...props} />,
                li: ({ node, ...props }: any) => <li className="" {...props} />,
                strong: ({ node, ...props }: any) => <strong className="text-[#f0eee6] font-semibold" {...props} />,
                hr: ({ node, ...props }: any) => <hr className="my-12 border-[#f0eee6]/5" {...props} />,
                a: ({ node, ...props }: any) => <a className="text-[#f0eee6] hover:text-white transition-colors" {...props} />,
              }}
            >
              {aboutContent}
            </ReactMarkdown>
          </div>

          {/* Dynamic Blog Posts Section */}
          {posts.length > 0 && (
            <div className="mt-24 border-t border-white/10 pt-16">
              <h2 className="text-3xl font-bold text-[#f0eee6] mb-12 flex items-center gap-3">
                <span className="text-2xl">📚</span>
                Latest Research Articles
              </h2>
              <div className="space-y-12">
                {posts.map((post) => (
                  <div key={post.slug} className="group relative">
                    <div className="flex items-center gap-4 text-sm text-[#f0eee6]/40 mb-3">
                      <span>{post.date}</span>
                    </div>
                    <Link href={`/blog/${post.slug}`} className="block">
                      <h3 className="text-2xl font-bold text-white mb-3">
                        {post.title}
                      </h3>
                    </Link>
                    <p className="text-[#f0eee6]/60 text-lg leading-relaxed mb-4">
                      {post.description}
                    </p>
                    <Link
                      href={`/blog/${post.slug}`}
                      className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 font-medium transition-colors"
                    >
                      Read full article
                      <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                      </svg>
                    </Link>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </>
  );
}

