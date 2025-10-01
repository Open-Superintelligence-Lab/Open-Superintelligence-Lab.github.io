'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import Image from 'next/image';
import 'highlight.js/styles/github-dark.css';
import 'katex/dist/katex.min.css';

interface MarkdownRendererProps {
  content: string;
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="prose prose-invert prose-lg max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeHighlight, rehypeKatex]}
        components={{
          // Custom heading styles
          h1: ({ children }) => (
            <h1 className="text-4xl font-bold mb-8 mt-12 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent leading-tight">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-3xl font-semibold mb-6 mt-10 text-gray-100 leading-tight">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-2xl font-semibold mb-4 mt-8 text-gray-200 leading-tight">
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-xl font-semibold mb-3 mt-6 text-gray-300 leading-tight">
              {children}
            </h4>
          ),
          // Custom paragraph styles
          p: ({ children }) => (
            <p className="text-gray-300 leading-relaxed mb-10 text-lg">
              {children}
            </p>
          ),
          // Custom link styles
          a: ({ href, children }) => (
            <a
              href={href}
              className="text-blue-400 hover:text-blue-300 underline transition-colors"
              target={href?.startsWith('http') ? '_blank' : undefined}
              rel={href?.startsWith('http') ? 'noopener noreferrer' : undefined}
            >
              {children}
            </a>
          ),
          // Custom list styles
          ul: ({ children }) => (
            <ul className="list-disc list-inside mb-10 space-y-4 text-gray-300 text-lg">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-inside mb-10 space-y-4 text-gray-300 text-lg">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="text-gray-300 ml-4 leading-relaxed">
              {children}
            </li>
          ),
          // Custom code block styles
          code: ({ className, children, ...props }) => {
            const inline = !className;
            return inline ? (
              <code
                className="bg-slate-800 text-orange-400 px-2 py-1 rounded text-sm font-mono"
                {...props}
              >
                {children}
              </code>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="bg-slate-900 border border-gray-700 rounded-lg p-6 overflow-x-auto mb-10 text-sm">
              {children}
            </pre>
          ),
          // Custom blockquote styles
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-blue-500 pl-6 italic text-gray-400 my-10 bg-slate-800/30 py-4 rounded-r-lg">
              {children}
            </blockquote>
          ),
          // Custom image handling with Next.js Image optimization
          img: ({ src, alt }) => {
            if (!src) return null;
            
            // Handle external images
            if (typeof src === 'string' && src.startsWith('http')) {
              return (
                <div className="my-6 rounded-lg overflow-hidden border border-gray-700">
                  <Image
                    src={src}
                    alt={alt || ''}
                    width={800}
                    height={400}
                    className="w-full h-auto"
                    loading="lazy"
                  />
                  {alt && (
                    <p className="text-center text-sm text-gray-400 mt-2 px-4 pb-4">
                      {alt}
                    </p>
                  )}
                </div>
              );
            }
            
            // Handle local images with Next.js Image
            return (
              <div className="my-6 rounded-lg overflow-hidden border border-gray-700">
                <Image
                  src={src as string}
                  alt={alt || ''}
                  width={800}
                  height={400}
                  className="w-full h-auto"
                  priority={false}
                />
                {alt && (
                  <p className="text-center text-sm text-gray-400 mt-2 px-4 pb-4">
                    {alt}
                  </p>
                )}
              </div>
            );
          },
          // Custom table styles
          table: ({ children }) => (
            <div className="overflow-x-auto my-8">
              <table className="min-w-full border border-gray-700 rounded-lg">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-slate-800">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-gray-700">
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-slate-800/50 transition-colors">
              {children}
            </tr>
          ),
          th: ({ children }) => (
            <th className="px-6 py-4 text-left text-gray-200 font-semibold text-lg">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-6 py-4 text-gray-300 text-lg">
              {children}
            </td>
          ),
          // Custom horizontal rule
          hr: () => (
            <hr className="my-12 border-gray-700" />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
