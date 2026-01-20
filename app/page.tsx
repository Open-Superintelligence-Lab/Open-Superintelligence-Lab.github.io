import fs from "fs";
import path from "path";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getAllPosts, BlogPost } from "@/lib/blog-utils";
import Link from "next/link";

export default function Home() {
  const aboutPath = path.join(process.cwd(), "ABOUT_LAB.md");
  const aboutContent = fs.readFileSync(aboutPath, "utf8");

  const researchPath = path.join(process.cwd(), "RESEARCH.md");
  const researchContent = fs.existsSync(researchPath) ? fs.readFileSync(researchPath, "utf8") : "";

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
                <span className="inline-block py-2 bg-gradient-to-b from-[#faf9f6] via-[#faf9f6] to-[#faf9f6]/60 bg-clip-text text-transparent">
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
                h1: ({ ...props }: any) => <h1 className="text-4xl font-bold text-[#faf9f6] mb-8 border-b border-[#faf9f6]/10 pb-4" {...props} />,
                h2: ({ ...props }: any) => <h2 className="text-2xl font-bold text-[#faf9f6] mt-12 mb-6" {...props} />,
                h3: ({ ...props }: any) => <h3 className="text-xl font-bold text-[#faf9f6] mt-8 mb-4 transition-colors" {...props} />,
                p: ({ ...props }: any) => <p className="text-[#faf9f6]/85 text-lg leading-relaxed mb-6" {...props} />,
                ul: ({ ...props }: any) => <ul className="list-disc list-inside mb-6 space-y-3 text-[#faf9f6]/85 text-lg" {...props} />,
                li: ({ ...props }: any) => <li className="" {...props} />,
                strong: ({ ...props }: any) => <strong className="text-[#faf9f6] font-semibold" {...props} />,
                hr: ({ ...props }: any) => <hr className="my-12 border-[#faf9f6]/5" {...props} />,
                a: ({ ...props }: any) => <a className="text-[#faf9f6] hover:text-white transition-colors" {...props} />,
                blockquote: ({ ...props }: any) => <blockquote className="border-l-4 border-[#faf9f6]/30 pl-6 my-8 italic text-[#faf9f6]/75" {...props} />,
              }}
            >
              {aboutContent}
            </ReactMarkdown>

            {researchContent && (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({ ...props }: any) => <h1 className="text-4xl font-bold text-[#faf9f6] mb-8 border-b border-[#faf9f6]/10 pb-4" {...props} />,
                  h2: ({ ...props }: any) => <h2 className="text-2xl font-bold text-[#faf9f6] mt-12 mb-6" {...props} />,
                  h3: ({ ...props }: any) => <h3 className="text-xl font-bold text-[#faf9f6] mt-8 mb-4 transition-colors" {...props} />,
                  p: ({ ...props }: any) => <p className="text-[#faf9f6]/85 text-lg leading-relaxed mb-6" {...props} />,
                  ul: ({ ...props }: any) => <ul className="list-disc list-inside mb-6 space-y-3 text-[#faf9f6]/85 text-lg" {...props} />,
                  li: ({ ...props }: any) => <li className="" {...props} />,
                  strong: ({ ...props }: any) => <strong className="text-[#faf9f6] font-semibold" {...props} />,
                  hr: ({ ...props }: any) => <hr className="my-12 border-[#faf9f6]/5" {...props} />,
                  a: ({ ...props }: any) => <a className="text-[#faf9f6] hover:text-white transition-colors" {...props} />,
                  blockquote: ({ ...props }: any) => <blockquote className="border-l-4 border-[#faf9f6]/30 pl-6 my-8 italic text-[#faf9f6]/75" {...props} />,
                }}
              >
                {researchContent}
              </ReactMarkdown>
            )}
          </div>

        </div>
      </main>
    </>
  );
}

