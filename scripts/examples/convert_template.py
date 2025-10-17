#!/usr/bin/env python3
"""
Template: Convert Paper Figures to PNG

Copy this file and customize for your paper.
Save as: convert_YOUR_PAPER.py
"""

from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from convert_pdf_figures import convert_figures

# ============================================================================
# CONFIGURATION - CUSTOMIZE THIS SECTION
# ============================================================================

# Paper slug (URL-friendly name)
PAPER_SLUG = "your-paper-name"

# Define paths
project_root = Path(__file__).parent.parent.parent
figures_dir = project_root / f"public/content/{PAPER_SLUG}/figures"  # or arXiv-*/figures
output_dir = project_root / f"public/content/{PAPER_SLUG}/images"

# Map PDF filenames to descriptive PNG names
# TIP: Organize by section/category for better clarity
figure_mapping = {
    # === Introduction / Overview ===
    "fig1.pdf": "problem-overview.png",
    "fig2.pdf": "proposed-solution.png",
    
    # === Method / Architecture ===
    "architecture.pdf": "system-architecture.png",
    "algorithm.pdf": "algorithm-diagram.png",
    
    # === Results ===
    "results_main.pdf": "main-results.png",
    "ablation.pdf": "ablation-study.png",
    
    # === Appendix ===
    "appendix_fig1.pdf": "detailed-analysis.png",
    
    # Add all your figures here...
}

# Optional: Custom settings
DPI = 300           # Resolution (150=low, 300=standard, 600=high)
CROP = True         # Remove whitespace
VERBOSE = True      # Show progress

# ============================================================================
# CONVERSION SCRIPT - NO NEED TO MODIFY
# ============================================================================

def main():
    print("="*60)
    print(f"Converting {PAPER_SLUG.upper()} Figures")
    print("="*60)
    print(f"Input:  {figures_dir}")
    print(f"Output: {output_dir}")
    print(f"Figures: {len(figure_mapping)}")
    print()
    
    # Check if input directory exists
    if not figures_dir.exists():
        print(f"❌ Error: Input directory not found: {figures_dir}")
        print(f"\nPlease:")
        print(f"  1. Update PAPER_SLUG variable")
        print(f"  2. Ensure figures are in: {figures_dir}")
        sys.exit(1)
    
    # Convert all figures
    converted, skipped, errors = convert_figures(
        input_dir=figures_dir,
        output_dir=output_dir,
        figure_mapping=figure_mapping,
        dpi=DPI,
        crop=CROP,
        verbose=VERBOSE
    )
    
    # Summary
    if errors == 0:
        print(f"\n✅ All {converted} figures converted successfully!")
        print(f"📁 Images saved to: {output_dir}")
        print(f"\n💡 Next steps:")
        print(f"   1. Check image quality in: {output_dir}")
        print(f"   2. Add images to your markdown content")
        print(f"   3. Reference as: /content/{PAPER_SLUG}/images/FILENAME.png")
    else:
        print(f"\n⚠️  Completed with {errors} error(s)")
        print(f"   ✓ Converted: {converted}")
        print(f"   ⚠ Skipped: {skipped}")
        print(f"   ✗ Errors: {errors}")

if __name__ == "__main__":
    main()

