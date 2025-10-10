'use client';

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useLanguage } from "@/components/providers/language-provider";
import { useState, useEffect, useRef } from "react";
import { getCourseModules } from "@/lib/course-structure";

interface LessonItem {
  title: string;
  titleZh: string;
  href: string;
}

interface ModuleData {
  title: string;
  titleZh: string;
  icon: React.ReactNode;
  lessons: LessonItem[];
}

export function CourseNavigation() {
  const { language } = useLanguage();
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const activeLinkRef = useRef<HTMLAnchorElement>(null);

  const modules = getCourseModules();

  // Auto-scroll to active lesson on mount and pathname change
  useEffect(() => {
    if (activeLinkRef.current) {
      activeLinkRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }
  }, [pathname]);

  const NavigationContent = () => (
    <>
      <div className="mb-6">
        <h3 className="text-lg font-bold text-white mb-2">
          {language === 'en' ? 'Course Contents' : '课程目录'}
        </h3>
        <p className="text-xs text-slate-400">
          {language === 'en' ? 'Navigate through the lessons' : '浏览课程内容'}
        </p>
      </div>

      <nav className="space-y-6">
        {modules.map((module, moduleIndex) => (
          <div key={moduleIndex}>
            <div className="flex items-center gap-2 mb-3">
              <div className="text-blue-400">
                {module.icon}
              </div>
              <h4 className="text-sm font-semibold text-slate-300">
                {language === 'en' ? module.title : module.titleZh}
              </h4>
            </div>
            <ul className="space-y-1 ml-7">
              {module.lessons.map((lesson, lessonIndex) => {
                const isActive = pathname === lesson.href;
                const lessonNumber = lessonIndex + 1;
                return (
                  <li key={lessonIndex}>
                    <Link
                      ref={isActive ? activeLinkRef : null}
                      href={lesson.href}
                      onClick={() => setIsOpen(false)}
                      className={`
                        block px-3 py-2 rounded-lg text-sm transition-all duration-200
                        ${isActive 
                          ? 'bg-blue-600/20 text-blue-400 font-medium border-l-2 border-blue-400' 
                          : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                        }
                      `}
                    >
                      <span className="font-semibold mr-2">{lessonNumber}.</span>
                      {language === 'en' ? lesson.title : lesson.titleZh}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      <div className="mt-8 pt-6 border-t border-slate-700">
        <Link
          href="/learn"
          className="flex items-center gap-2 text-sm text-slate-400 hover:text-blue-400 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
          </svg>
          {language === 'en' ? 'Course Home' : '课程首页'}
        </Link>
      </div>
    </>
  );

  return (
    <>
      {/* Mobile Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="lg:hidden fixed top-20 left-4 z-50 bg-slate-800 border border-slate-700 p-3 rounded-lg shadow-lg hover:bg-slate-700 transition-colors"
        aria-label="Toggle course navigation"
      >
        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Mobile Sidebar */}
      <aside
        className={`
          lg:hidden fixed top-0 left-0 bottom-0 w-80 bg-slate-900 border-r border-slate-700 z-50 transform transition-transform duration-300 ease-in-out overflow-y-auto
          ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        <div className="p-6">
          <button
            onClick={() => setIsOpen(false)}
            className="absolute top-4 right-4 text-slate-400 hover:text-white"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          <NavigationContent />
        </div>
      </aside>

      {/* Desktop Sidebar */}
      <aside className="hidden lg:block fixed left-0 top-0 bottom-0 w-80 bg-slate-900/50 backdrop-blur-sm border-r border-slate-700 overflow-y-auto pt-20">
        <div className="p-6">
          <NavigationContent />
        </div>
      </aside>
    </>
  );
}

