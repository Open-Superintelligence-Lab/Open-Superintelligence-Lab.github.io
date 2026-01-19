import React from 'react';
import fs from 'fs';
import path from 'path';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function RequestReviewPage() {
    const filePath = path.join(process.cwd(), 'REVIEW_CRITERIA.md');
    const content = fs.readFileSync(filePath, 'utf8');

    return (
        <div className="min-h-screen bg-[#1f1e1d] text-[#f0eee6]/80 pt-32 pb-24">
            <div className="container mx-auto px-6 max-w-4xl">
                <div className="markdown-content">
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                            h1: ({ node, ...props }) => <h1 className="text-4xl font-bold text-[#f0eee6] mb-8 border-b border-[#f0eee6]/10 pb-4" {...props} />,
                            h2: ({ node, ...props }) => <h2 className="text-2xl font-bold text-[#f0eee6] mt-12 mb-6" {...props} />,
                            h3: ({ node, ...props }) => <h3 className="text-xl font-bold text-[#f0eee6] mt-8 mb-4 hover:text-blue-400 transition-colors" {...props} />,
                            p: ({ node, ...props }) => <p className="text-lg leading-relaxed mb-6" {...props} />,
                            ul: ({ node, ...props }) => <ul className="list-disc list-inside mb-6 space-y-3 ml-4 text-lg" {...props} />,
                            li: ({ node, ...props }) => <li className="" {...props} />,
                            strong: ({ node, ...props }) => <strong className="text-[#f0eee6] font-semibold" {...props} />,
                            hr: ({ node, ...props }) => <hr className="my-12 border-[#f0eee6]/5" {...props} />,
                        }}
                    >
                        {content}
                    </ReactMarkdown>
                </div>
            </div>
        </div>
    );
}
