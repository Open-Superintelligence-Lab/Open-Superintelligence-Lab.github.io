import { getPostBySlug, getAllPosts } from "@/lib/blog-utils";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Link from "next/link";

interface PageProps {
    params: {
        slug: string;
    };
}

export async function generateStaticParams() {
    const posts = getAllPosts();
    return posts.map((post) => ({
        slug: post.slug,
    }));
}

export default async function BlogPostPage({ params }: PageProps) {
    const { slug } = params;
    const post = getPostBySlug(slug);

    if (!post) {
        // If the post is not found in the markdown files, 
        // it might be one of the "old" hardcoded blogs.
        // However, if we reached here, it means no static folder matched.
        notFound();
    }

    return (
        <div className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]/90 pt-32 pb-24">
            <div className="container mx-auto px-6 max-w-4xl">
                <div className="mb-12">


                    <div className="flex items-center gap-4 text-sm text-[#faf9f6]/60 mb-4">
                        <span>{post.date}</span>
                    </div>

                    <h1 className="text-4xl md:text-5xl font-bold text-[#faf9f6] mb-6 leading-tight">
                        {post.title}
                    </h1>

                    <p className="text-xl text-[#faf9f6]/75 leading-relaxed border-l-4 border-blue-500/50 pl-6 my-8 italic">
                        {post.description}
                    </p>

                    {post.youtubeId && (
                        <div className="my-12 aspect-video w-full overflow-hidden rounded-2xl border border-white/10 shadow-2xl">
                            <iframe
                                width="100%"
                                height="100%"
                                src={`https://www.youtube.com/embed/${post.youtubeId}`}
                                title="YouTube video player"
                                frameBorder="0"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                                allowFullScreen
                            ></iframe>
                        </div>
                    )}
                </div>

                <div className="markdown-content">
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                            h1: ({ node, ...props }) => <h1 className="text-4xl font-bold text-[#faf9f6] mb-8 border-b border-[#faf9f6]/10 pb-4" {...props} />,
                            h2: ({ node, ...props }) => <h2 className="text-2xl font-bold text-[#faf9f6] mt-12 mb-6" {...props} />,
                            h3: ({ node, ...props }) => <h3 className="text-xl font-bold text-[#faf9f6] mt-8 mb-4 hover:text-blue-400 transition-colors" {...props} />,
                            p: ({ node, ...props }) => <p className="text-lg leading-relaxed mb-6" {...props} />,
                            ul: ({ node, ...props }) => <ul className="list-disc list-inside mb-6 space-y-3 ml-4 text-lg" {...props} />,
                            li: ({ node, ...props }) => <li className="" {...props} />,
                            strong: ({ node, ...props }) => <strong className="text-[#faf9f6] font-semibold" {...props} />,
                            hr: ({ node, ...props }) => <hr className="my-12 border-[#faf9f6]/5" {...props} />,
                            blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-blue-500/50 pl-4 py-1 italic text-[#faf9f6]/75 my-6" {...props} />,
                            a: ({ node, ...props }) => <a className="text-blue-400 hover:text-blue-300 transition-colors" target="_blank" rel="noopener noreferrer" {...props} />,
                            code: ({ node, ...props }) => (
                                <code className="bg-[#2a2928] px-1.5 py-0.5 rounded text-blue-300 font-mono text-sm" {...props} />
                            ),
                            pre: ({ node, ...props }) => (
                                <pre className="bg-[#2a2928] p-6 rounded-xl border border-white/5 overflow-x-auto my-8 font-mono text-sm leading-relaxed" {...props} />
                            ),
                            img: ({ node, ...props }) => (
                                <span className="block my-12">
                                    <img
                                        className="rounded-2xl border border-white/10 shadow-2xl mx-auto"
                                        {...props}
                                    />
                                    {props.alt && (
                                        <span className="block text-center text-sm text-[#faf9f6]/50 mt-4 italic">
                                            {props.alt}
                                        </span>
                                    )}
                                </span>
                            ),
                        }}
                    >
                        {post.content}
                    </ReactMarkdown>
                </div>

                <div className="mt-24 pt-8 border-t border-white/10">
                    <div className="flex justify-between items-center">
                        <div className="text-sm text-[#faf9f6]/50">
                            © 2026 Open Superintelligence Lab
                        </div>
                        <div className="flex gap-6">
                            <a href="https://x.com/open-superintelligence-lab" className="text-[#faf9f6]/50 hover:text-blue-400 transition-colors">Twitter</a>
                            <a href="https://github.com/open-superintelligence-lab" className="text-[#faf9f6]/50 hover:text-blue-400 transition-colors">GitHub</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
