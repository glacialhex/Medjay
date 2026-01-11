#!/usr/bin/env python3
"""
Combine Egyptian_hieroglyphs and Glyphnet datasets for better coverage
"""
import os
import shutil
from pathlib import Path

# Source paths
EGYPTIAN_HIEROGLYPHS = Path("datasets/Egyptian_hieroglyphs/Dataset/train")
GLYPHNET = Path("datasets/Glyphnet/Automated_extracted/Automated")
OUTPUT_DIR = Path("datasets/Combined_Hieroglyphs")

def combine_datasets():
    """Combine multiple hieroglyph datasets into one"""
    
    # Create output directory
    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    
    # Copy Egyptian_hieroglyphs dataset
    print("Processing Egyptian_hieroglyphs...")
    if EGYPTIAN_HIEROGLYPHS.exists():
        for class_folder in EGYPTIAN_HIEROGLYPHS.iterdir():
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name
            dest_folder = OUTPUT_DIR / "train" / class_name
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            for img_file in class_folder.glob("*"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    shutil.copy2(img_file, dest_folder / f"egyptian_{img_file.name}")
                    total_images += 1
        
        print(f"  Copied {total_images} images from Egyptian_hieroglyphs")
    
    # Copy Glyphnet dataset
    print("Processing Glyphnet...")
    if GLYPHNET.exists():
        for class_folder in GLYPHNET.iterdir():
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name
            dest_folder = OUTPUT_DIR / "train" / class_name
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            copied = 0
            for img_file in class_folder.glob("*"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    shutil.copy2(img_file, dest_folder / f"glyphnet_{img_file.name}")
                    copied += 1
                    total_images += 1
            
            if copied > 0:
                print(f"  {class_name}: +{copied} images")
    
    print(f"\n✅ Combined dataset created:")
    print(f"   Total images: {total_images}")
    print(f"   Location: {OUTPUT_DIR}/train")
    print(f"   Classes: {len(list((OUTPUT_DIR / 'train').iterdir()))}")

if __name__ == "__main__":
    combine_datasets()
