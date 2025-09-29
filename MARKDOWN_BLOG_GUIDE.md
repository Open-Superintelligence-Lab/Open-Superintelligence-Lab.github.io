# Markdown Blog Post Guide

This guide explains how to create beautiful, image-rich research blog posts using Markdown for your research pages.

## Quick Start

### 1. Create Your Markdown File

Place your markdown file in the `/public` directory with a descriptive name:

```
/public/content-research-[topic]-[subtopic].md
```

Example: `/public/content-research-deepseek-v3-2-exp.md`

### 2. Write Your Content

Use standard Markdown syntax with enhanced styling:

```markdown
# Main Title

![Hero Image](https://via.placeholder.com/1200x600/1e293b/60a5fa?text=Your+Title)

## Introduction

Your introduction text here...

### Subsection

More content with **bold** and *italic* text.

> Important blockquote or callout

## Code Examples

\`\`\`python
def example():
    print("Hello, World!")
\`\`\`

## Tables

| Feature | Value |
|---------|-------|
| Speed   | Fast  |
| Cost    | Low   |
```

### 3. Create Your Page Component

Create a new page in `/app/research/[category]/[topic]/page.tsx`:

```typescript
'use client';

import Link from "next/link";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { useState, useEffect } from "react";

export default function YourResearchPage() {
  const [markdownContent, setMarkdownContent] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/content-research-your-topic.md')
      .then(res => res.text())
      .then(content => {
        setMarkdownContent(content);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading markdown:', error);
        setLoading(false);
      });
  }, []);

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20"></div>
        
        <div className="relative container mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Your Research Title
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              Brief description of your research
            </p>
          </div>
        </div>
      </section>

      <main className="container mx-auto px-6 py-12">
        <div className="max-w-4xl mx-auto">
          {loading ? (
            <div className="flex justify-center items-center py-20">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400"></div>
            </div>
          ) : (
            <article className="bg-slate-900/50 border border-gray-800 rounded-lg p-8 md:p-12">
              <MarkdownRenderer content={markdownContent} />
            </article>
          )}
        </div>
      </main>
    </>
  );
}
```

## Markdown Features

### Headings

```markdown
# H1 - Main Title (blue-purple gradient)
## H2 - Section (gray-100)
### H3 - Subsection (gray-200)
#### H4 - Minor heading (gray-300)
```

### Text Formatting

```markdown
**Bold text**
*Italic text*
`Inline code`
[Link text](https://example.com)
```

### Images

#### External Images (Recommended for Flexibility)

```markdown
![Alt text](https://via.placeholder.com/1200x600/1e293b/60a5fa?text=Your+Title)
```

#### Local Images

1. Place images in `/public/images/research/[category]/`
2. Reference them in markdown:

```markdown
![Alt text](/images/research/deepseek/architecture.png)
```

### Code Blocks

````markdown
```python
def hello_world():
    print("Hello, World!")
```
````

Supported languages (with syntax highlighting):
- Python
- JavaScript/TypeScript
- Bash
- JSON
- And many more (via highlight.js)

### Tables

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |
```

### Blockquotes

```markdown
> Important note or callout
> Can span multiple lines
```

### Lists

#### Unordered Lists
```markdown
- Item 1
- Item 2
  - Nested item
- Item 3
```

#### Ordered Lists
```markdown
1. First item
2. Second item
3. Third item
```

### Horizontal Rule

```markdown
---
```

## Image Guidelines

### Recommended Image Sizes

- **Hero Images**: 1200×600px
- **Section Images**: 800×400px
- **Inline Images**: 600×300px
- **Diagrams**: Variable, but aim for 800-1000px width

### Placeholder Images

For development, use placeholder services:

```markdown
![Description](https://via.placeholder.com/WIDTH×HEIGHT/BG_COLOR/TEXT_COLOR?text=Your+Text)
```

Example:
```markdown
![Architecture](https://via.placeholder.com/1200x600/1e293b/60a5fa?text=System+Architecture)
```

### Image Optimization Tips

1. **Use WebP format** for smaller file sizes
2. **Compress images** before uploading (aim for <200KB)
3. **Provide descriptive alt text** for accessibility
4. **Use external CDNs** for large images when possible

## Styling Customization

All markdown is rendered through the `MarkdownRenderer` component. To customize styles:

1. Edit `/components/markdown-renderer.tsx`
2. Modify the component styles in the `components` prop
3. Add custom CSS classes as needed

### Color Palette

The default theme uses:
- **Primary**: Blue (#60a5fa) to Purple (#a78bfa)
- **Background**: Slate-900 (#0f172a)
- **Text**: Gray-300 (#d1d5db)
- **Borders**: Gray-700/800
- **Code**: Orange-400 (#fb923c) on Slate-800

## Advanced Features

### Frontmatter (Optional)

Add metadata at the top of your markdown:

```markdown
---
title: Your Research Title
date: 2025-09-29
author: Your Name
tags: AI, Research, DeepSeek
---

# Your Content Starts Here
```

Use the `parseFrontmatter` utility from `/lib/markdown-utils.ts` to extract this data.

### Table of Contents

Use the `extractHeadings` or `generateTableOfContents` functions from `/lib/markdown-utils.ts`:

```typescript
import { extractHeadings } from '@/lib/markdown-utils';

const headings = extractHeadings(markdownContent);
// Use headings to build a custom TOC
```

### Reading Time

```typescript
import { getReadingTime } from '@/lib/markdown-utils';

const minutes = getReadingTime(markdownContent);
// Display: "5 min read"
```

## Best Practices

1. **Structure**: Use clear heading hierarchy (H1 → H2 → H3)
2. **Readability**: Keep paragraphs concise (3-5 sentences)
3. **Visuals**: Include images every 2-3 sections
4. **Code**: Add syntax highlighting language for all code blocks
5. **Links**: Open external links in new tabs (handled automatically)
6. **Alt Text**: Always provide descriptive alt text for images
7. **Testing**: Preview your content in development mode before deploying

## Example Structure

```markdown
# Main Title

![Hero Image](...)

## Introduction

Brief overview of the research...

## Background

Context and motivation...

### Problem Statement

What problem are we solving?

## Methodology

How we approached it...

### Implementation

Technical details...

```python
# Code example
```

## Results

What we found...

| Metric | Before | After |
|--------|--------|-------|
| Speed  | 100ms  | 50ms  |

## Discussion

Analysis of results...

> Key insight or takeaway

## Conclusion

Summary and future work...

---

## References

1. [Link to paper](...)
2. [Link to code](...)
```

## Troubleshooting

### Markdown Not Loading

- Check the file path in `fetch()`
- Verify the file is in `/public`
- Check browser console for errors

### Images Not Displaying

- Verify image URLs are correct
- For local images, ensure they're in `/public`
- Check image file extensions

### Styling Issues

- Clear your browser cache
- Check if syntax highlighting CSS is loaded
- Verify custom CSS doesn't conflict

### Build Errors

- Ensure all imports are correct
- Check for TypeScript errors
- Verify markdown syntax is valid

## Resources

- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Flavored Markdown](https://github.github.com/gfm/)
- [Highlight.js Languages](https://highlightjs.org/static/demo/)
- [Placeholder Image Service](https://placeholder.com/)

## Need Help?

Check the example at `/app/research/deepseek/v3-2-exp/page.tsx` for a complete working implementation.
