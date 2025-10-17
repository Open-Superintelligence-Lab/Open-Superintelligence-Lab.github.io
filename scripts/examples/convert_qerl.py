#!/usr/bin/env python3
"""
Example: Convert QeRL paper figures to PNG

This is a reference implementation showing how to convert
academic paper figures for blog posts.
"""

from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from convert_pdf_figures import convert_figures

# Define paths relative to project root
project_root = Path(__file__).parent.parent.parent
figures_dir = project_root / "public/content/qerl-quantization-reinforcement-learning/arXiv-2510.11696v1/figures"
output_dir = project_root / "public/content/qerl-quantization-reinforcement-learning/images"

# Map all PDF figures to meaningful output names
# Organize by category for clarity
figure_mapping = {
    # === Main Framework ===
    "framework4.pdf": "qerl-framework.png",
    "noise_merge.pdf": "noise-merge-diagram.png",
    
    # === Core Results ===
    "performance.png": "performance.png",  # Already PNG, will be cropped
    "entropy_v2.pdf": "entropy-exploration.png",
    "da_gr_v2.pdf": "reward-growth.png",
    "decay_curve_v2.pdf": "noise-schedule.png",
    
    # === Ablation Studies ===
    "rank_ablation_v2.pdf": "rank-ablation.png",
    "scheduler_ablation_v2.pdf": "scheduler-ablation.png",
    "rank_speed_v2.pdf": "rank-speed.png",
    
    # === Model Comparisons ===
    "appendix_7B.pdf": "7b-results.png",
    "appendix_32B.pdf": "32b-results.png",
    "appendix_lr_lora.pdf": "lr-lora-comparison.png",
    "appendix_lr_qerl.pdf": "lr-qerl-comparison.png",
    
    # === Detailed Analysis ===
    "app_entropy.pdf": "entropy-appendix.png",
    "entropy_abs_line_v2.pdf": "entropy-absolute.png",
    "fig7_1_v2.pdf": "memory-comparison.png",
    "fig7_2_v2.pdf": "speed-comparison.png",
}

def main():
    print("="*60)
    print("Converting QeRL Paper Figures")
    print("="*60)
    print(f"Input:  {figures_dir}")
    print(f"Output: {output_dir}")
    print(f"Figures: {len(figure_mapping)}")
    print()
    
    # Convert all figures
    converted, skipped, errors = convert_figures(
        input_dir=figures_dir,
        output_dir=output_dir,
        figure_mapping=figure_mapping,
        dpi=300,        # High-quality for web
        crop=True,      # Remove whitespace
        verbose=True    # Show progress
    )
    
    # Summary
    if errors == 0:
        print(f"\n✅ All {converted} figures converted successfully!")
        print(f"📁 Images saved to: {output_dir}")
    else:
        print(f"\n⚠️  Completed with {errors} error(s)")
        print(f"   Converted: {converted}")
        print(f"   Skipped: {skipped}")

if __name__ == "__main__":
    main()


