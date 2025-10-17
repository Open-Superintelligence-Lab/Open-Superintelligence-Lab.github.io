#!/usr/bin/env python3
"""
Universal PDF Figure to PNG Converter

This script converts academic paper PDF figures to high-quality PNG images
with automatic whitespace cropping. Perfect for blog posts and documentation.

Features:
- Uses PyMuPDF for clean, high-quality rendering
- Automatic whitespace detection and cropping
- Configurable output names via mapping dictionary
- 300 DPI equivalent resolution
- Optimized PNG compression

Author: Open Superintelligence Lab
License: MIT
"""

import fitz  # PyMuPDF
from PIL import Image
import io
from pathlib import Path
import sys
import argparse


def crop_whitespace(image, threshold=250, border=10):
    """
    Crop whitespace from PIL Image.
    
    Args:
        image: PIL Image object
        threshold: Pixel brightness threshold (0-255). Pixels darker than this are kept.
        border: Number of pixels to add around the cropped content
    
    Returns:
        Cropped PIL Image
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get pixel data
    pixels = image.load()
    width, height = image.size
    
    # Find bounding box
    left, top, right, bottom = width, height, 0, 0
    
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            # If pixel is not white-ish
            if r < threshold or g < threshold or b < threshold:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)
    
    # Add border
    left = max(0, left - border)
    top = max(0, top - border)
    right = min(width, right + border)
    bottom = min(height, bottom + border)
    
    # Crop
    if left < right and top < bottom:
        return image.crop((left, top, right, bottom))
    return image


def convert_pdf_to_png(pdf_path, output_path, dpi=300, crop=True):
    """
    Convert a single PDF to PNG.
    
    Args:
        pdf_path: Path to input PDF
        output_path: Path to output PNG
        dpi: Resolution (dots per inch)
        crop: Whether to crop whitespace
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc[0]  # Get first page
        
        # Calculate zoom factor for desired DPI
        # Default is 72 DPI, so zoom = desired_dpi / 72
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Crop whitespace if requested
        if crop:
            img = crop_whitespace(img)
        
        # Save
        img.save(output_path, 'PNG', optimize=True)
        
        doc.close()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def convert_png_copy(png_path, output_path, crop=True):
    """
    Copy and optionally crop an existing PNG.
    
    Args:
        png_path: Path to input PNG
        output_path: Path to output PNG
        crop: Whether to crop whitespace
    
    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(png_path)
        
        if crop:
            img = crop_whitespace(img)
        
        img.save(output_path, 'PNG', optimize=True, dpi=(300, 300))
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def convert_figures(input_dir, output_dir, figure_mapping, dpi=300, crop=True, verbose=True):
    """
    Convert multiple PDF figures to PNG.
    
    Args:
        input_dir: Directory containing input PDFs
        output_dir: Directory for output PNGs
        figure_mapping: Dict mapping input filenames to output filenames
        dpi: Resolution for conversion
        crop: Whether to crop whitespace
        verbose: Whether to print progress
    
    Returns:
        Tuple of (converted_count, skipped_count, error_count)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Converting figures from: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Found {len(figure_mapping)} figures to process\n")
    
    converted = 0
    skipped = 0
    errors = 0
    
    for input_name, output_name in figure_mapping.items():
        input_path = input_dir / input_name
        output_path = output_dir / output_name
        
        if not input_path.exists():
            if verbose:
                print(f"⚠️  Skipping {input_name} (not found)")
            skipped += 1
            continue
        
        # Handle PNG files
        if input_name.endswith('.png'):
            success = convert_png_copy(input_path, output_path, crop=crop)
            if success:
                if verbose:
                    size_kb = output_path.stat().st_size / 1024
                    print(f"✓ Cropped {input_name} -> {output_name} ({size_kb:.1f} KB)")
                converted += 1
            else:
                if verbose:
                    print(f"✗ Error processing {input_name}")
                errors += 1
            continue
        
        # Handle PDF files
        success = convert_pdf_to_png(input_path, output_path, dpi=dpi, crop=crop)
        if success:
            if verbose:
                size_kb = output_path.stat().st_size / 1024
                print(f"✓ Converted {input_name} -> {output_name} ({size_kb:.1f} KB)")
            converted += 1
        else:
            if verbose:
                print(f"✗ Error converting {input_name}")
            errors += 1
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✨ Conversion complete!")
        print(f"   ✓ Converted: {converted}")
        print(f"   ⚠ Skipped: {skipped}")
        print(f"   ✗ Errors: {errors}")
        print(f"{'='*60}")
    
    return converted, skipped, errors


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Convert PDF figures to PNG with automatic whitespace cropping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single PDF
  python convert_pdf_figures.py input.pdf output.png
  
  # Convert with custom DPI
  python convert_pdf_figures.py input.pdf output.png --dpi 150
  
  # Convert without cropping
  python convert_pdf_figures.py input.pdf output.png --no-crop

For batch conversion, import the script as a module and use convert_figures().
        """
    )
    
    parser.add_argument('input', help='Input PDF file')
    parser.add_argument('output', help='Output PNG file')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI (default: 300)')
    parser.add_argument('--no-crop', action='store_true', help='Disable whitespace cropping')
    parser.add_argument('--threshold', type=int, default=250, 
                       help='Whitespace threshold 0-255 (default: 250)')
    parser.add_argument('--border', type=int, default=10,
                       help='Border pixels around cropped image (default: 10)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting: {input_path.name}")
    print(f"Output: {output_path}")
    print(f"DPI: {args.dpi}")
    print(f"Crop: {not args.no_crop}\n")
    
    success = convert_pdf_to_png(
        input_path, 
        output_path, 
        dpi=args.dpi, 
        crop=not args.no_crop
    )
    
    if success:
        size_kb = output_path.stat().st_size / 1024
        print(f"✓ Success! Output size: {size_kb:.1f} KB")
        sys.exit(0)
    else:
        print("✗ Conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()


