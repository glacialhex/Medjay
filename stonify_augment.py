"""
Stone Texture Augmentation for YOLO Training

This script takes the clean hieroglyph training images and creates
augmented versions that simulate real stone wall conditions:
- Stone texture overlay
- Random shadows/lighting
- Noise (grain)
- Low contrast
- Color tinting (sandstone/limestone colors)
"""

import cv2
import numpy as np
import os
from pathlib import Path
import random

# Input/Output paths
INPUT_DIR = Path("datasets/hieroglyph-yolo-dataset/train/images")
OUTPUT_DIR = Path("datasets/hieroglyph-yolo-dataset-stone/train/images")
LABELS_INPUT = Path("datasets/hieroglyph-yolo-dataset/train/labels")
LABELS_OUTPUT = Path("datasets/hieroglyph-yolo-dataset-stone/train/labels")

# Stone-like colors (RGB)
STONE_COLORS = [
    (212, 196, 172),  # Sandstone
    (199, 186, 168),  # Limestone
    (176, 161, 142),  # Weathered stone
    (220, 210, 190),  # Light sand
    (160, 145, 125),  # Dark stone
]


def create_stone_texture(shape):
    """Generate procedural stone-like texture"""
    h, w = shape[:2]
    
    # Create base noise at multiple scales
    noise = np.zeros((h, w), dtype=np.float32)
    
    for scale in [4, 8, 16, 32]:
        small = np.random.rand(h // scale + 1, w // scale + 1).astype(np.float32)
        resized = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        noise += resized / scale
    
    # Normalize
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def add_stone_effect(image):
    """Apply stone texture and color to an image"""
    h, w = image.shape[:2]
    
    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    
    # 1. Apply stone color tint
    stone_color = random.choice(STONE_COLORS)
    stone_color_norm = np.array(stone_color[::-1], dtype=np.float32) / 255.0  # BGR
    
    # Blend with stone color (make the white areas look like stone)
    tinted = img_float * 0.6 + stone_color_norm * 0.4
    
    # 2. Add texture
    texture = create_stone_texture(image.shape)
    texture_strength = random.uniform(0.05, 0.15)
    tinted = tinted + (texture[:, :, np.newaxis] - 0.5) * texture_strength
    
    # 3. Add shadow gradient (simulate uneven lighting)
    if random.random() > 0.5:
        gradient = np.linspace(0.7, 1.0, w).reshape(1, -1)
        if random.random() > 0.5:
            gradient = gradient[:, ::-1]
        tinted = tinted * gradient[:, :, np.newaxis]
    
    # 4. Add some noise (grain)
    noise_strength = random.uniform(0.02, 0.08)
    noise = np.random.randn(h, w, 3).astype(np.float32) * noise_strength
    tinted = tinted + noise
    
    # 5. Reduce contrast (simulate worn/faded carvings)
    contrast = random.uniform(0.6, 0.9)
    mean = tinted.mean()
    tinted = (tinted - mean) * contrast + mean
    
    # 6. Add slight blur (simulate erosion)
    if random.random() > 0.7:
        blur_size = random.choice([3, 5])
        tinted = cv2.GaussianBlur(tinted, (blur_size, blur_size), 0)
    
    # Clip and convert back
    tinted = np.clip(tinted * 255, 0, 255).astype(np.uint8)
    
    return tinted


def augment_dataset():
    """Process all images in the dataset"""
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(INPUT_DIR.glob("*.png")) + list(INPUT_DIR.glob("*.jpg"))
    
    print(f"Found {len(image_files)} images to augment")
    
    for i, img_path in enumerate(image_files):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read: {img_path}")
            continue
        
        # Apply stone effect
        augmented = add_stone_effect(img)
        
        # Save augmented image
        output_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(output_path), augmented)
        
        # Copy label file (bounding boxes stay the same)
        label_path = LABELS_INPUT / (img_path.stem + ".txt")
        if label_path.exists():
            label_output = LABELS_OUTPUT / label_path.name
            with open(label_path, 'r') as f:
                content = f.read()
            with open(label_output, 'w') as f:
                f.write(content)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_files)}")
    
    print(f"\nDone! Augmented images saved to {OUTPUT_DIR}")
    print(f"Labels copied to {LABELS_OUTPUT}")
    
    # Create data.yaml for the new dataset
    data_yaml = f"""
path: {OUTPUT_DIR.parent.absolute()}
train: train/images
val: train/images

names:
  0: hieroglyph
"""
    
    yaml_path = OUTPUT_DIR.parent / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(data_yaml.strip())
    
    print(f"Data config saved to {yaml_path}")


if __name__ == "__main__":
    augment_dataset()
