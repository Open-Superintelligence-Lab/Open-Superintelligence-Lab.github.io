#!/usr/bin/env python3
"""
Extract figures from The Art of Scaling RL paper
"""

from pathlib import Path
import sys
import fitz  # PyMuPDF
from PIL import Image
import io

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from convert_pdf_figures import crop_whitespace

# Paths
project_root = Path(__file__).parent.parent.parent
paper_path = project_root / "public/content/art-of-scaling-rl/paper.pdf"
output_dir = project_root / "public/content/art-of-scaling-rl/images"

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

def extract_images_from_pdf(pdf_path, output_dir, min_width=200, min_height=200):
    """
    Extract all images from a PDF paper
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save images
        min_width: Minimum image width to extract
        min_height: Minimum image height to extract
    """
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    image_count = 0
    extracted_images = []
    
    print(f"Total pages: {len(doc)}\n")
    
    # Iterate through pages
    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"Processing page {page_num + 1}...")
        
        # Get images on this page
        image_list = page.get_images()
        
        if image_list:
            print(f"  Found {len(image_list)} image(s)")
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]  # XREF number
            
            try:
                # Extract image
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Load image with PIL
                img = Image.open(io.BytesIO(image_bytes))
                width, height = img.size
                
                # Skip small images (likely icons, logos, etc.)
                if width < min_width or height < min_height:
                    print(f"  Skipping small image: {width}x{height}")
                    continue
                
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Crop whitespace
                img = crop_whitespace(img, threshold=250, border=10)
                
                # Generate filename
                image_count += 1
                output_path = output_dir / f"figure_{image_count:02d}_page_{page_num + 1}.png"
                
                # Save
                img.save(output_path, 'PNG', optimize=True, dpi=(300, 300))
                
                size_kb = output_path.stat().st_size / 1024
                print(f"  ✓ Extracted: {output_path.name} ({width}x{height}, {size_kb:.1f} KB)")
                
                extracted_images.append({
                    'filename': output_path.name,
                    'page': page_num + 1,
                    'size': (width, height),
                    'path': output_path
                })
                
            except Exception as e:
                print(f"  ✗ Error extracting image: {e}")
    
    doc.close()
    
    print(f"\n{'='*60}")
    print(f"✨ Extraction complete!")
    print(f"   Total images extracted: {image_count}")
    print(f"   Saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    # Print renaming guide
    if extracted_images:
        print("📝 RENAMING GUIDE:")
        print("Review the extracted images and rename them descriptively.")
        print("Example mapping:\n")
        for img in extracted_images[:5]:  # Show first 5 as examples
            print(f"  {img['filename']} (page {img['page']}) -> [descriptive-name].png")
        if len(extracted_images) > 5:
            print(f"  ... and {len(extracted_images) - 5} more images")
    
    return extracted_images

if __name__ == "__main__":
    if not paper_path.exists():
        print(f"❌ Error: Paper not found at {paper_path}")
        sys.exit(1)
    
    print("="*60)
    print("Extracting Figures from The Art of Scaling RL")
    print("="*60)
    print()
    
    extracted_images = extract_images_from_pdf(paper_path, output_dir)
    
    print("\n💡 Next steps:")
    print("   1. Review extracted images")
    print("   2. Rename files to descriptive names")
    print("   3. Add to markdown: ![Description](/content/art-of-scaling-rl/images/filename.png)")

