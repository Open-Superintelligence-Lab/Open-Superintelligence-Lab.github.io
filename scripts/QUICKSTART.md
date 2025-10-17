# Quick Start Guide

## Convert Figures for a New Blog Post

### 1. Install Dependencies (One Time)

```bash
pip install PyMuPDF pillow
```

### 2. Prepare Your Paper Figures

Download paper source from arXiv (if available):

```bash
# Example for arXiv paper 2510.11696
cd public/content/your-paper-slug/
wget https://arxiv.org/e-print/2510.11696
tar -xzf 2510.11696
```

Or manually place PDF figures in:
```
public/content/your-paper-slug/figures/
```

### 3. Create Conversion Script

Copy the template:

```bash
cp scripts/examples/convert_template.py scripts/examples/convert_YOUR_PAPER.py
```

Edit the new file and update:

```python
# Change this
PAPER_SLUG = "your-paper-name"

# And map your figures
figure_mapping = {
    "fig1.pdf": "architecture-diagram.png",
    "fig2.pdf": "results-comparison.png",
    # ... add all figures
}
```

### 4. Run Conversion

```bash
python scripts/examples/convert_YOUR_PAPER.py
```

### 5. Use in Blog Post

In your markdown file:

```markdown
![System Architecture](/content/your-paper-slug/images/architecture-diagram.png)
*Figure 1: Overview of the proposed system architecture.*
```

## Command Line Usage (Single Files)

Quick conversion of a single PDF:

```bash
# Basic conversion
python scripts/convert_pdf_figures.py input.pdf output.png

# High resolution
python scripts/convert_pdf_figures.py input.pdf output.png --dpi 600

# No cropping
python scripts/convert_pdf_figures.py input.pdf output.png --no-crop
```

## Common Issues

**Issue**: Images have too much whitespace
**Solution**: Check if `crop=True` in your script

**Issue**: Figures too small
**Solution**: Increase DPI (e.g., `dpi=600`)

**Issue**: Files too large
**Solution**: Decrease DPI (e.g., `dpi=150`) for simple charts

**Issue**: "Module not found"
**Solution**: `pip install PyMuPDF pillow`

## Examples

See working examples in `scripts/examples/`:
- `convert_qerl.py` - Full example with 17 figures
- `convert_template.py` - Template for new papers

## Need Help?

Check the full documentation: [scripts/README.md](README.md)

