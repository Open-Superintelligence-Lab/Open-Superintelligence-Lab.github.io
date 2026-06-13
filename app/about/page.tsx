import fs from "fs";
import path from "path";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function AboutPage() {
  const filePath = path.join(process.cwd(), "ABOUT_LAB.md");
  const content = fs.readFileSync(filePath, "utf8");

  return (
    <div className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]/90 pt-32 pb-24">
      <div className="container mx-auto max-w-4xl px-6">
        <div className="mb-12">
          <div className="mb-4 flex items-center gap-4 text-sm text-[#faf9f6]/60">
            <span>Open Superintelligence Lab</span>
            <span>About</span>
          </div>
          <h1 className="mb-6 text-4xl font-bold leading-tight text-[#faf9f6] md:text-5xl">
            About
          </h1>
        </div>

        <div className="markdown-content">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              p: ({ ...props }) => (
                <p className="mb-6 text-lg leading-relaxed text-[#faf9f6]/85" {...props} />
              ),
            }}
          >
            {content}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
