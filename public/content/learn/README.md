# Course Content Structure

This directory contains markdown files and images for the AI/ML course lessons.

## Directory Structure

```
learn/
├── math/
│   ├── derivatives/
│   │   ├── derivatives-content.md
│   │   ├── derivative-graph.png (placeholder - add your image here)
│   │   └── tangent-line.png (placeholder - add your image here)
│   └── functions/
│       ├── functions-content.md
│       ├── linear-function.png (add your image here)
│       ├── relu-function.png (add your image here)
│       └── function-composition.png (add your image here)
└── neural-networks/
    ├── introduction/
    │   ├── introduction-content.md
    │   ├── neural-network-diagram.png (add your image here)
    │   ├── layer-types.png (add your image here)
    │   ├── training-process.png (add your image here)
    │   └── depth-vs-performance.png (add your image here)
    ├── forward-propagation/
    │   ├── forward-propagation-content.md
    │   ├── forward-prop-diagram.png (add your image here)
    │   ├── forward-example.png (add your image here)
    │   ├── activations-comparison.png (add your image here)
    │   └── matrix-backprop.png (add your image here)
    ├── backpropagation/
    │   ├── backpropagation-content.md
    │   ├── backprop-overview.png (add your image here)
    │   ├── backprop-steps.png (add your image here)
    │   └── matrix-backprop.png (add your image here)
    └── training/
        ├── training-content.md
        ├── training-loop.png (add your image here)
        ├── gradient-descent.png (add your image here)
        ├── gd-variants.png (add your image here)
        ├── optimizers-comparison.png (add your image here)
        ├── lr-schedules.png (add your image here)
        └── training-curves.png (add your image here)
```

## How to Add Images

1. Place your PNG/JPG images in the corresponding lesson folder
2. Reference them in the markdown using:
   ```markdown
   ![Alt Text](image-name.png)
   ```
3. The images will be served from `/content/learn/[lesson-path]/[image-name]`

## Markdown Frontmatter Format

Each lesson markdown file should start with frontmatter:

```markdown
---
hero:
  title: "Lesson Title"
  subtitle: "Lesson Subtitle"
  tags:
    - "📐 Category"
    - "⏱️ Reading Time"
---

# Your content here...
```

## Adding New Lessons

1. Create a new folder under the appropriate category
2. Add a `{folder-name}-content.md` file
3. Add your images
4. Create a page component in `app/learn/[category]/[lesson-name]/page.tsx`:

```tsx
import { LessonPage } from "@/components/lesson-page";

export default function YourLessonPage() {
  return (
    <LessonPage
      contentPath="category/lesson-name"
      prevLink={{ href: "/previous", label: "← Previous" }}
      nextLink={{ href: "/next", label: "Next →" }}
    />
  );
}
```

