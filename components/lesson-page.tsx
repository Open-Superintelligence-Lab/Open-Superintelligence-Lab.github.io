"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { CourseNavigation } from "@/components/course-navigation";
import { useEffect, useState } from "react";
import { getAdjacentLessons } from "@/lib/course-structure";

interface HeroData {
  title: string;
  subtitle: string;
  tags: string[];
}

interface LessonPageProps {
  contentPath: string;
  prevLink?: { href: string; label: string };
  nextLink?: { href: string; label: string };
  youtubeVideoId?: string;
}

export function LessonPage({ contentPath, prevLink, nextLink, youtubeVideoId }: LessonPageProps) {
  const pathname = usePathname();
  const [markdownContent, setMarkdownContent] = useState<string>('');
  const [heroData, setHeroData] = useState<HeroData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Auto-determine next/prev links from course structure if not provided
  const adjacentLessons = getAdjacentLessons(pathname);
  const effectivePrevLink = prevLink || (adjacentLessons.prev ? {
    href: adjacentLessons.prev.href,
    label: `← Previous: ${adjacentLessons.prev.title}`
  } : undefined);

  const effectiveNextLink = nextLink || (adjacentLessons.next ? {
    href: adjacentLessons.next.href,
    label: `Next: ${adjacentLessons.next.title} →`
  } : undefined);

  useEffect(() => {
    const fetchMarkdownContent = async () => {
      try {
        const response = await fetch(`/content/learn/${contentPath}/${contentPath.split('/').pop()}-content.md`);
        const content = await response.text();

        // Parse frontmatter
        const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
        if (frontmatterMatch) {
          const frontmatterContent = frontmatterMatch[1];
          const markdownBody = frontmatterMatch[2];

          // Default hero data
          const heroData: HeroData = {
            title: "",
            subtitle: "",
            tags: []
          };

          // Extract values from frontmatter
          const lines = frontmatterContent.split('\n');
          let currentKey = '';
          let currentArray: string[] = [];

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (trimmedLine.startsWith('hero:')) continue;

            if (trimmedLine.includes(':')) {
              const [key, ...valueParts] = trimmedLine.split(':');
              const value = valueParts.join(':').trim().replace(/^["']|["']$/g, '');

              switch (key.trim()) {
                case 'title':
                  heroData.title = value;
                  break;
                case 'subtitle':
                  heroData.subtitle = value;
                  break;
                case 'tags':
                  currentKey = 'tags';
                  currentArray = [];
                  break;
              }
            } else if (trimmedLine.startsWith('- ')) {
              if (currentKey === 'tags') {
                const tagValue = trimmedLine.substring(2).replace(/^["']|["']$/g, '');
                currentArray.push(tagValue);
              }
            } else if (trimmedLine === '' && currentArray.length > 0) {
              if (currentKey === 'tags') {
                heroData.tags = currentArray;
                currentArray = [];
                currentKey = '';
              }
            }
          }

          // Handle final array
          if (currentArray.length > 0 && currentKey === 'tags') {
            heroData.tags = currentArray;
          }

          setHeroData(heroData);
          setMarkdownContent(markdownBody);
        } else {
          setMarkdownContent(content);
        }
      } catch (error) {
        console.error('Failed to fetch markdown content:', error);
        setMarkdownContent('# Error loading content\n\nFailed to load the lesson content.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchMarkdownContent();
  }, [contentPath]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4"></div>
          <p className="text-slate-400">Loading lesson...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Course Navigation Sidebar */}
      <CourseNavigation />

      {/* Main Content with Sidebar Offset */}
      <div className="lg:ml-80">
        {/* Hero Section */}
        <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 via-purple-600/10 to-blue-600/10"></div>

          <div className="relative container mx-auto px-6 pt-24 pb-12">
            <div className="max-w-4xl mx-auto">
              {/* Back to Course */}
              <Link
                href="/learn"
                className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 mb-8 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Back to Course
              </Link>

              <div className="relative">
                <h1 className="text-4xl md:text-5xl font-bold mb-4 leading-tight">
                  <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                    {heroData?.title || 'Lesson'}
                  </span>
                </h1>
                <p className="text-xl text-slate-400 mb-6">
                  {heroData?.subtitle || ''}
                </p>

                {/* Tags */}
                {heroData?.tags && heroData.tags.length > 0 && (
                  <div className="flex items-center gap-3 text-sm text-slate-400">
                    {heroData.tags.map((tag, index) => (
                      <span key={index} className="flex items-center gap-2">
                        {index > 0 && <span className="text-slate-600">•</span>}
                        <span>{tag}</span>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Main Content */}
        <main className="bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 min-h-screen">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 pt-8 pb-16">
            <article className="max-w-4xl mx-auto">
              <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl shadow-2xl p-8 sm:p-12">
                {youtubeVideoId && (
                  <div className="mb-8">
                    <div className="relative" style={{ paddingTop: '56.25%' }}>
                      <iframe
                        className="absolute top-0 left-0 w-full h-full rounded-lg shadow-2xl"
                        src={`https://www.youtube.com/embed/${youtubeVideoId}`}
                        title="YouTube video player"
                        frameBorder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                      ></iframe>
                    </div>
                  </div>
                )}
                <div className="prose prose-lg prose-invert max-w-none">
                  <MarkdownRenderer content={markdownContent} />
                </div>
              </div>

              {/* Navigation */}
              <div className="mt-12 flex flex-col sm:flex-row items-center justify-between gap-4">
                {effectivePrevLink ? (
                  <Link
                    href={effectivePrevLink.href}
                    className="group flex items-center gap-2 px-6 py-3 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-500/50 text-slate-300 hover:text-blue-400 font-medium rounded-xl transition-all duration-300"
                  >
                    <svg className="w-5 h-5 group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                    {effectivePrevLink.label}
                  </Link>
                ) : (
                  <div></div>
                )}

                {effectiveNextLink ? (
                  <Link
                    href={effectiveNextLink.href}
                    className="group flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-xl transition-all duration-300"
                  >
                    {effectiveNextLink.label}
                    <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </Link>
                ) : (
                  <Link
                    href="/learn"
                    className="group flex items-center gap-2 px-6 py-3 bg-emerald-600 hover:bg-emerald-700 text-white font-medium rounded-xl transition-all duration-300"
                  >
                    Course Complete! 🎉
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </Link>
                )}
              </div>
            </article>
          </div>
        </main>
      </div>
    </>
  );
}

