/* eslint-disable @typescript-eslint/no-unused-vars */
import React from 'react';
import fs from 'fs';
import path from 'path';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function PublishPage() {
    const filePath = path.join(process.cwd(), 'PUBLISH_GUIDE.md');
    const content = fs.readFileSync(filePath, 'utf8');

    return (
        <div className="min-h-screen bg-[#1f1e1d] text-[#faf9f6]/90 pt-32 pb-24">
            <div className="container mx-auto px-6 max-w-4xl">
                <div className="markdown-content">
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                            h1: ({ ...props }) => <h1 className="text-4xl font-bold text-[#faf9f6] mb-8 border-b border-[#faf9f6]/10 pb-4" {...props} />,
                            h2: ({ ...props }) => <h2 className="text-2xl font-bold text-[#faf9f6] mt-12 mb-6" {...props} />,
                            h3: ({ ...props }) => <h3 className="text-xl font-bold text-[#faf9f6] mt-8 mb-4 hover:text-blue-400 transition-colors" {...props} />,
                            p: ({ ...props }) => <p className="text-lg leading-relaxed mb-6" {...props} />,
                            ul: ({ ...props }) => <ul className="list-disc list-inside mb-6 space-y-3 ml-4 text-lg" {...props} />,
                            li: ({ ...props }) => <li className="" {...props} />,
                            strong: ({ ...props }) => <strong className="text-[#faf9f6] font-semibold" {...props} />,
                            hr: ({ ...props }) => <hr className="my-12 border-[#faf9f6]/5" {...props} />,
                            blockquote: ({ ...props }) => <blockquote className="border-l-4 border-blue-500/50 pl-4 py-1 italic text-[#faf9f6]/75 my-6" {...props} />,
                            a: ({ ...props }) => <a className="text-blue-400 hover:text-blue-300 transition-colors" target="_blank" rel="noopener noreferrer" {...props} />,
                        }}
                    >
                        {content}
                    </ReactMarkdown>
                </div>
            </div>
        </div>
    );
}
