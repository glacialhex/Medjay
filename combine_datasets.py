#!/usr/bin/env python3
"""
Combine Egyptian_hieroglyphs and Glyphnet datasets for better coverage

Glyphnet structure: Images are in numbered folders but class is in filename
  e.g., Manual_extracted/Manual/Preprocessed/20/200145_I9.png -> class I9

Egyptian_hieroglyphs structure: Organized by class folders
  e.g., Dataset/train/A55/image.png -> class A55
"""
import shutil
import re
from pathlib import Path

# Source paths
EGYPTIAN_HIEROGLYPHS = Path("datasets/Egyptian_hieroglyphs/Dataset/train")
# Glyphnet has multiple sources - we'll scan all of them
GLYPHNET_BASE = Path("datasets/Glyphnet")
GLYPHNET_SOURCES = [
    GLYPHNET_BASE / "Manual_extracted" / "Manual" / "Preprocessed",
    GLYPHNET_BASE / "Manual_extracted" / "Manual" / "Raw",
    GLYPHNET_BASE / "Automated_extracted" / "Automated" / "Preprocessed",
    GLYPHNET_BASE / "Automated_extracted" / "Automated" / "Raw",
]
OUTPUT_DIR = Path("datasets/Combined_Hieroglyphs")


def extract_class_from_filename(filename):
    """
    Extract Gardiner class from Glyphnet filename.
    Examples:
      200145_I9.png -> I9
      030029_G43.png -> G43
      200315_UNKNOWN.png -> UNKNOWN (skip these)
    """
    # Pattern: digits_CLASS.ext
    match = re.match(r'\d+_([A-Za-z0-9]+)\.\w+$', filename)
    if match:
        return match.group(1)
    return None


def combine_datasets():
    """Combine multiple hieroglyph datasets into one"""
    
    # Create output directory
    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    egyptian_count = 0
    glyphnet_count = 0
    skipped_unknown = 0

    # Copy Egyptian_hieroglyphs dataset (organized by class folders)
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
                    egyptian_count += 1

        print(f"  ✓ Copied {egyptian_count} images from Egyptian_hieroglyphs")
    else:
        print(f"  ⚠️ Egyptian_hieroglyphs not found at: {EGYPTIAN_HIEROGLYPHS}")

    # Copy Glyphnet dataset (class in filename, not folder)
    print("\nProcessing Glyphnet...")
    for glyphnet_source in GLYPHNET_SOURCES:
        if not glyphnet_source.exists():
            print(f"  - Skipping (not found): {glyphnet_source}")
            continue

        print(f"  Processing: {glyphnet_source.relative_to(GLYPHNET_BASE)}")
        source_count = 0

        # Glyphnet has numbered subfolders (3, 5, 7, 9, 20, 21, etc.)
        for subfolder in glyphnet_source.iterdir():
            if not subfolder.is_dir():
                continue

            for img_file in subfolder.glob("*"):
                if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                    continue

                # Extract class from filename
                class_name = extract_class_from_filename(img_file.name)
                if class_name is None or class_name.upper() == "UNKNOWN":
                    skipped_unknown += 1
                    continue

                dest_folder = OUTPUT_DIR / "train" / class_name
                dest_folder.mkdir(parents=True, exist_ok=True)

                # Create unique filename using source folder info
                source_id = glyphnet_source.name[:4]  # "Prep" or "Raw_"
                unique_name = f"glyphnet_{source_id}_{img_file.name}"

                dest_path = dest_folder / unique_name
                if not dest_path.exists():  # Avoid duplicates
                    shutil.copy2(img_file, dest_path)
                    source_count += 1
                    total_images += 1
                    glyphnet_count += 1

        print(f"    ✓ Added {source_count} images")

    if skipped_unknown > 0:
        print(f"\n  ⚠️ Skipped {skipped_unknown} images with UNKNOWN class")

    # Print summary
    num_classes = len(list((OUTPUT_DIR / 'train').iterdir())) if (OUTPUT_DIR / 'train').exists() else 0

    print(f"\n{'='*50}")
    print(f"✅ Combined dataset created:")
    print(f"   Egyptian_hieroglyphs: {egyptian_count:,} images")
    print(f"   Glyphnet: {glyphnet_count:,} images")
    print(f"   Total images: {total_images:,}")
    print(f"   Classes: {num_classes}")
    print(f"   Location: {OUTPUT_DIR}/train")
    print(f"{'='*50}")

    if total_images < 10000:
        print(f"\n⚠️ WARNING: Only {total_images:,} images combined. Expected 21,000+")
        print("   Check that both datasets are downloaded correctly.")
    else:
        print(f"\n🎉 Success! Dataset is ready for training.")
        print("   Next step: python3 train_classifier.py")


if __name__ == "__main__":
    combine_datasets()
