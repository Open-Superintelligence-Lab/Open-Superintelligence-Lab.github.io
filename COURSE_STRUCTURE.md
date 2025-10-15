# Course Structure Documentation

## Overview

The learning course has been reorganized to use **markdown files** stored in the `public/content/learn/` directory, following the same pattern as the blog posts. This makes it easy to manage content and add images.

## 📁 File Structure

```
public/content/learn/
├── README.md                    # Documentation for content management
├── math/
│   ├── functions/
│   │   ├── functions-content.md
│   │   └── [add your images here]
│   ├── derivatives/
│   │   ├── derivatives-content.md
│   │   ├── derivative-graph.png
│   │   └── tangent-line.png
│   ├── vectors/
│   │   ├── vectors-content.md
│   │   └── [images included]
│   ├── matrices/
│   │   ├── matrices-content.md
│   │   └── [images included]
│   └── gradients/
│       ├── gradients-content.md
│       └── [images included]
└── neural-networks/
    ├── introduction/
    │   ├── introduction-content.md
    │   └── [add your images here]
    ├── forward-propagation/
    │   ├── forward-propagation-content.md
    │   └── [add your images here]
    ├── backpropagation/
    │   ├── backpropagation-content.md
    │   └── [add your images here]
    └── training/
        ├── training-content.md
        └── [add your images here]
```

## 🎓 Course Modules

### Module 1: Mathematics Fundamentals

1. **Functions** (`/learn/math/functions`)
   - Linear functions
   - Activation functions (Sigmoid, ReLU, Tanh)
   - Loss functions
   - Why non-linearity matters

2. **Derivatives** (`/learn/math/derivatives`)
   - What derivatives are
   - Why they matter in AI
   - Common derivative rules
   - Practical examples with loss functions

3. **Vectors** (`/learn/math/vectors`)
   - What vectors are (magnitude and direction)
   - Vector components and representation
   - Vector operations (addition, scalar multiplication)
   - Applications in machine learning

4. **Matrices** (`/learn/math/matrices`)
   - Matrix fundamentals
   - Matrix operations (multiplication, transpose)
   - Matrix transformations
   - Role in neural networks

5. **Gradients** (`/learn/math/gradients`)
   - Understanding gradients
   - Partial derivatives
   - Gradient computation
   - Gradient descent in optimization

### Module 2: Neural Networks from Scratch

1. **Introduction** (`/learn/neural-networks/introduction`)
   - What neural networks are
   - Basic architecture (input, hidden, output layers)
   - How they learn
   - Real-world applications

2. **Forward Propagation** (`/learn/neural-networks/forward-propagation`)
   - The forward pass process
   - Weighted sums and activations
   - Step-by-step numerical examples
   - Matrix operations

3. **Backpropagation** (`/learn/neural-networks/backpropagation`)
   - The backpropagation algorithm
   - Chain rule in action
   - Gradient computation
   - Common challenges (vanishing/exploding gradients)

4. **Training & Optimization** (`/learn/neural-networks/training`)
   - Gradient descent variants (SGD, mini-batch, batch)
   - Advanced optimizers (Adam, RMSprop, Momentum)
   - Hyperparameters and learning rate schedules
   - Best practices and common pitfalls

## 🛠️ Technical Implementation

### Components Created

1. **LessonPage Component** (`components/lesson-page.tsx`)
   - Reusable component that loads markdown content
   - Handles frontmatter parsing
   - Supports navigation between lessons
   - Similar to blog post structure

2. **Page Routes** (`app/learn/...`)
   - Each lesson has a simple page component
   - Uses `LessonPage` with configuration
   - Clean and maintainable

### How It Works

1. **Markdown files** are stored in `public/content/learn/[category]/[lesson]/`
2. Each file has **frontmatter** with hero data (title, subtitle, tags)
3. **Images** are placed alongside the markdown files
4. **Page components** load the markdown using the `LessonPage` component
5. Images are referenced as `![alt](image.png)` and served from `/content/learn/...`

### Example Markdown Frontmatter

```markdown
---
hero:
  title: "Derivatives"
  subtitle: "The Foundation of Neural Network Training"
  tags:
    - "📐 Mathematics"
    - "⏱️ 10 min read"
---

# Your content here...

![Derivative Graph](derivative-graph.png)
```

## 📝 Adding New Content

### To Add a New Lesson:

1. **Create folder structure:**
   ```bash
   mkdir -p public/content/learn/[category]/[lesson-name]
   ```

2. **Create markdown file:**
   ```bash
   touch public/content/learn/[category]/[lesson-name]/[lesson-name]-content.md
   ```

3. **Add frontmatter and content** to the markdown file

4. **Add images** to the same folder

5. **Create page component:**
   ```tsx
   // app/learn/[category]/[lesson-name]/page.tsx
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

## 🖼️ Adding Images

### Placeholder Images Currently Referenced:

**Math - Derivatives:**
- `derivative-graph.png` - Visual showing derivative as slope
- `tangent-line.png` - Tangent line illustration

**Math - Functions:**
- `linear-function.png` - Linear function visualization
- `relu-function.png` - ReLU activation graph
- `function-composition.png` - Function composition diagram

**Neural Networks - Introduction:**
- `neural-network-diagram.png` - Basic NN architecture
- `layer-types.png` - Input, hidden, output layers
- `training-process.png` - Training loop diagram
- `depth-vs-performance.png` - Network depth impact

**Neural Networks - Forward Propagation:**
- `forward-prop-diagram.png` - Data flow diagram
- `forward-example.png` - Example calculation
- `activations-comparison.png` - Different activation functions
- `matrix-backprop.png` - Matrix operations

**Neural Networks - Backpropagation:**
- `backprop-overview.png` - Algorithm overview
- `backprop-steps.png` - Step-by-step process
- `matrix-backprop.png` - Matrix form backprop

**Neural Networks - Training:**
- `training-loop.png` - Training loop visualization
- `gradient-descent.png` - Gradient descent illustration
- `gd-variants.png` - GD variants comparison
- `optimizers-comparison.png` - Optimizer behaviors
- `lr-schedules.png` - Learning rate schedules
- `training-curves.png` - Loss/accuracy curves

### To Add Your Images:

1. Create your images (PNG or JPG recommended)
2. Place them in the appropriate lesson folder
3. They're already referenced in the markdown - just replace the placeholders!

## 🎨 Design Features

- **Beautiful gradient backgrounds** matching the site theme
- **Syntax highlighting** for code blocks
- **Responsive design** for mobile and desktop
- **Navigation** between lessons with prev/next buttons
- **Markdown rendering** with support for:
  - Headings, paragraphs, lists
  - Code blocks
  - Images
  - Tables
  - Math formulas (using KaTeX in MarkdownRenderer)

## 🚀 Next Steps

1. **Add your images** - Replace placeholder PNG files with actual visualizations
2. **Expand content** - Add more lessons or modules as needed
3. **Test on localhost** - Visit `/learn` to see the course
4. **Customize styling** - Adjust colors/gradients in the components if desired

## 📋 Summary

✅ Course structure created with 9 lessons (5 math + 4 neural networks)  
✅ Markdown files in `public/content/learn/`  
✅ Reusable `LessonPage` component  
✅ Images ready for math lessons (vectors, matrices, gradients)  
✅ Navigation between lessons  
✅ Frontmatter support for hero sections  
✅ README documentation in content folder  

Your course is ready with comprehensive math fundamentals! 🎉

