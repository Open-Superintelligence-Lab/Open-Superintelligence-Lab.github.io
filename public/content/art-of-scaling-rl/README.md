# The Art of Scaling RL - Blog Post Assets

This directory contains all assets for the "Art of Scaling Reinforcement Learning Compute for LLMs" blog post.

## Directory Structure

```
art-of-scaling-rl/
├── README.md                          # This file
├── FIGURES_REFERENCE.md               # Complete guide to all 27 figures
├── paper.pdf                          # Original research paper
├── art-of-scaling-rl-content.md      # Blog post markdown content
└── images/                            # 27 extracted and renamed figures
    ├── scaling-law-overview.png
    ├── scalerl-framework-diagram.png
    ├── comprehensive-results-table.png
    └── ... (24 more figures)
```

## Blog Post Files

### Main Content
- **art-of-scaling-rl-content.md** - The complete blog post content with:
  - Abstract and introduction
  - Key research findings
  - ScaleRL recipe details
  - Experimental methodology
  - Practical implications
  - Surprising discoveries
  - Future directions

### Page Component
- **../../app/blog/art-of-scaling-rl/page.tsx** - Next.js page component

## Figures (27 total)

All figures have been:
✅ Extracted from the PDF using PyMuPDF
✅ Automatically cropped to remove whitespace
✅ Renamed with descriptive, meaningful names
✅ Optimized for web (PNG format, ~3MB total)

See `FIGURES_REFERENCE.md` for the complete list and usage examples.

## Key Figures to Use

### Essential (Must Include)
1. `scaling-law-overview.png` - Main concept visualization
2. `scalerl-framework-diagram.png` - Methodology diagram
3. `comprehensive-results-table.png` - Empirical validation
4. `recipe-asymptote-comparison.png` - Key finding: recipe matters
5. `math-reasoning-scaling.png` - Real-world example

### Important Supporting
6. `kl-divergence-dynamics.png` - Training stability
7. `loss-aggregation-ablation.png` - Design choices impact
8. `scaling-prediction-accuracy.png` - Predictive power

## Development Scripts

Scripts used to generate these assets (in `../../scripts/examples/`):

1. **extract_art_of_scaling_rl.py** - Extract all images from PDF
2. **rename_art_of_scaling_rl_figures.py** - Rename to descriptive names

## Usage in Blog Post

To add figures to the markdown content:

```markdown
![Scaling Law Overview](/content/art-of-scaling-rl/images/scaling-law-overview.png)
*Figure 1: Overview of RL scaling laws showing sigmoidal performance curves*
```

## Blog Post Status

✅ Content written (comprehensive 400+ lines)
✅ Page component created
✅ Added to homepage
✅ All 27 figures extracted and renamed
✅ Figures reference guide created
📝 Ready to add figures to content markdown

## Next Steps

1. Review `FIGURES_REFERENCE.md` to see all available figures
2. Add selected figures to `art-of-scaling-rl-content.md` at appropriate sections
3. Build and preview the blog post locally
4. Deploy to production

## Page URLs

- **Live URL:** https://opensuperintelligencelab.com/blog/art-of-scaling-rl/
- **Local:** http://localhost:3000/blog/art-of-scaling-rl/

## Research Paper

- **Title:** The Art of Scaling Reinforcement Learning Compute for LLMs
- **Authors:** Anonymous (under review)
- **Compute:** 400,000 GPU-hours of experiments
- **Key Contribution:** First systematic study with predictive RL scaling laws

---

Created: October 17, 2024
Last Updated: October 17, 2024

