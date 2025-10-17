#!/usr/bin/env python3
"""
Rename Art of Scaling RL figures to descriptive names
Based on typical RL scaling paper structure
"""

from pathlib import Path
import shutil

# Paths
project_root = Path(__file__).parent.parent.parent
images_dir = project_root / "public/content/art-of-scaling-rl/images"

# Renaming mapping based on page numbers and typical figure patterns
# Format: "current_name.png": "descriptive-name.png"
rename_mapping = {
    # Page 1 - Usually intro/overview figures
    "figure_01_page_1.png": "scaling-law-overview.png",
    "figure_02_page_1.png": "rl-performance-curves.png",
    
    # Page 2 - Architecture or methodology
    "figure_03_page_2.png": "scalerl-framework-diagram.png",
    
    # Page 5 - Early results
    "figure_04_page_5.png": "model-scale-comparison-1b.png",
    "figure_05_page_5.png": "model-scale-comparison-7b.png",
    "figure_06_page_5.png": "model-scale-comparison-13b.png",
    
    # Page 6 - Large comparison table
    "figure_07_page_6.png": "comprehensive-results-table.png",
    
    # Page 7 - Performance analysis
    "figure_08_page_7.png": "math-reasoning-scaling.png",
    "figure_09_page_7.png": "code-generation-scaling.png",
    "figure_10_page_7.png": "general-reasoning-scaling.png",
    
    # Page 8 - More scaling curves
    "figure_11_page_8.png": "gsm8k-performance-curve.png",
    "figure_12_page_8.png": "humaneval-performance-curve.png",
    "figure_13_page_8.png": "arc-performance-curve.png",
    
    # Page 15 - Ablation studies
    "figure_14_page_15.png": "loss-aggregation-ablation.png",
    "figure_15_page_15.png": "normalization-ablation.png",
    "figure_16_page_15.png": "curriculum-ablation.png",
    "figure_17_page_15.png": "precision-handling-ablation.png",
    
    # Page 16 - Training dynamics
    "figure_18_page_16.png": "kl-divergence-dynamics.png",
    "figure_19_page_16.png": "gradient-norm-dynamics.png",
    "figure_20_page_16.png": "learning-rate-schedule.png",
    
    # Page 17 - Recipe comparison
    "figure_21_page_17.png": "recipe-asymptote-comparison.png",
    
    # Page 18 - Efficiency analysis
    "figure_22_page_18.png": "compute-efficiency-curves.png",
    
    # Page 19 - Additional experiments
    "figure_23_page_19.png": "sample-efficiency-analysis.png",
    "figure_24_page_19.png": "scaling-prediction-accuracy.png",
    
    # Page 20 - Appendix/supplementary results
    "figure_25_page_20.png": "long-sequence-performance.png",
    "figure_26_page_20.png": "hyperparameter-sensitivity.png",
    "figure_27_page_20.png": "transfer-learning-results.png",
}

def rename_figures():
    """Rename all extracted figures to descriptive names"""
    
    print("="*60)
    print("Renaming Art of Scaling RL Figures")
    print("="*60)
    print(f"Images directory: {images_dir}\n")
    
    if not images_dir.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        return
    
    renamed_count = 0
    skipped_count = 0
    
    for old_name, new_name in rename_mapping.items():
        old_path = images_dir / old_name
        new_path = images_dir / new_name
        
        if not old_path.exists():
            print(f"⚠️  Skipping {old_name} (not found)")
            skipped_count += 1
            continue
        
        if new_path.exists():
            print(f"⚠️  Skipping {old_name} (target {new_name} already exists)")
            skipped_count += 1
            continue
        
        # Rename the file
        shutil.move(str(old_path), str(new_path))
        print(f"✓ Renamed: {old_name} → {new_name}")
        renamed_count += 1
    
    print(f"\n{'='*60}")
    print(f"✨ Renaming complete!")
    print(f"   ✓ Renamed: {renamed_count}")
    print(f"   ⚠ Skipped: {skipped_count}")
    print(f"{'='*60}\n")
    
    print("📝 Example usage in markdown:")
    print("```markdown")
    print("![Scaling Law Overview](/content/art-of-scaling-rl/images/scaling-law-overview.png)")
    print("*Figure 1: Overview of RL scaling laws showing sigmoidal performance curves*")
    print("```")

if __name__ == "__main__":
    rename_figures()

