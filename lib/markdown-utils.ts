/**
 * Utility functions for working with markdown blog posts
 */

export interface BlogPost {
  slug: string;
  title: string;
  description: string;
  date: string;
  tags: string[];
  content: string;
}

/**
 * Fetch markdown content from the public folder
 * @param path - Path to the markdown file relative to /public
 * @returns Promise resolving to the markdown content
 */
export async function fetchMarkdownContent(path: string): Promise<string> {
  try {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`Failed to fetch markdown: ${response.statusText}`);
    }
    return await response.text();
  } catch (error) {
    console.error('Error fetching markdown content:', error);
    throw error;
  }
}

/**
 * Parse frontmatter from markdown content
 * @param content - Raw markdown content with optional frontmatter
 * @returns Object containing frontmatter data and content
 */
export function parseFrontmatter(content: string): {
  data: Record<string, any>;
  content: string;
} {
  const frontmatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
  const match = content.match(frontmatterRegex);

  if (!match) {
    return { data: {}, content };
  }

  const [, frontmatter, markdownContent] = match;
  const data: Record<string, any> = {};

  // Parse YAML-like frontmatter
  frontmatter.split('\n').forEach(line => {
    const colonIndex = line.indexOf(':');
    if (colonIndex > -1) {
      const key = line.slice(0, colonIndex).trim();
      const value = line.slice(colonIndex + 1).trim();
      
      // Remove quotes if present
      data[key] = value.replace(/^["']|["']$/g, '');
    }
  });

  return { data, content: markdownContent };
}

/**
 * Get reading time estimate for markdown content
 * @param content - Markdown content
 * @param wordsPerMinute - Average reading speed (default: 200)
 * @returns Reading time in minutes
 */
export function getReadingTime(content: string, wordsPerMinute: number = 200): number {
  const words = content.trim().split(/\s+/).length;
  const minutes = Math.ceil(words / wordsPerMinute);
  return minutes;
}

/**
 * Extract all headings from markdown content
 * @param content - Markdown content
 * @returns Array of heading objects with level and text
 */
export function extractHeadings(content: string): Array<{ level: number; text: string; id: string }> {
  const headingRegex = /^(#{1,6})\s+(.+)$/gm;
  const headings: Array<{ level: number; text: string; id: string }> = [];
  
  let match;
  while ((match = headingRegex.exec(content)) !== null) {
    const level = match[1].length;
    const text = match[2];
    const id = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
    
    headings.push({ level, text, id });
  }
  
  return headings;
}

/**
 * Generate a table of contents from markdown content
 * @param content - Markdown content
 * @returns HTML string for table of contents
 */
export function generateTableOfContents(content: string): string {
  const headings = extractHeadings(content);
  
  if (headings.length === 0) return '';
  
  let toc = '<nav class="table-of-contents">\n<ul>\n';
  
  headings.forEach(heading => {
    const indent = '  '.repeat(heading.level - 1);
    toc += `${indent}<li><a href="#${heading.id}">${heading.text}</a></li>\n`;
  });
  
  toc += '</ul>\n</nav>';
  
  return toc;
}
