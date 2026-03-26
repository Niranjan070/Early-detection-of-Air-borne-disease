"""
Merge Dataset Script
====================
Merges your existing magnaporthe_oryzae dataset with the new Curvularia genus dataset.

What this script does:
1. Converts segmentation labels (polygons) → bounding box labels (YOLO detection format)
2. Remaps class IDs so pyricularia becomes magnaporthe_oryzae (class 0)
3. Copies all images and labels into a single merged dataset
4. Creates a new data.yaml for training

Final class mapping:
  0: magnaporthe_oryzae  (= pyricularia, the Rice Blast spore) ← YOUR TARGET
  1: alternaria
  2: bipolaris
  3: curvularia
  4: curvularia_eragrostidis
  5: exserohilum
  6: fusarium
  7: fusarium_microconidie
  8: mycelium
"""

import os
import shutil
import yaml
from pathlib import Path


# ─── CONFIGURATION ───
PROJECT_ROOT = Path(__file__).parent.parent

# Existing dataset (your original magnaporthe_oryzae data)
EXISTING_DATASET = PROJECT_ROOT / "data" / "splits"

# New downloaded dataset (Curvularia genus from Roboflow)
NEW_DATASET = PROJECT_ROOT / "data" / "new_dataset"

# Output merged dataset
MERGED_DATASET = PROJECT_ROOT / "data" / "merged"

# New class names (magnaporthe_oryzae first = class 0, your target!)
MERGED_CLASSES = [
    "magnaporthe_oryzae",       # 0 - Rice Blast spore (YOUR TARGET)
    "alternaria",               # 1
    "bipolaris",                # 2
    "curvularia",               # 3
    "curvularia_eragrostidis",  # 4
    "exserohilum",              # 5
    "fusarium",                 # 6
    "fusarium_microconidie",    # 7
    "mycelium",                 # 8
]

# Mapping: old class IDs in new dataset → merged class IDs
# Old: 0=alternaria, 1=bipolaris, 2=curvularia, 3=curvularia eragrostidis,
#      4=exserohilum, 5=fusarium, 6=fusarium microconidie, 7=mycelium, 8=pyricularia
NEW_TO_MERGED = {
    0: 1,   # alternaria → 1
    1: 2,   # bipolaris → 2
    2: 3,   # curvularia → 3
    3: 4,   # curvularia eragrostidis → 4
    4: 5,   # exserohilum → 5
    5: 6,   # fusarium → 6
    6: 7,   # fusarium microconidie → 7
    7: 8,   # mycelium → 8
    8: 0,   # pyricularia → 0 (magnaporthe_oryzae = same thing!)
}

# Existing dataset: class 0 stays as class 0 (magnaporthe_oryzae)
EXISTING_TO_MERGED = {
    0: 0,   # magnaporthe_oryzae → 0
}


def seg_to_bbox(coords):
    """
    Convert segmentation polygon coordinates to bounding box.
    
    Input:  list of floats [x1, y1, x2, y2, x3, y3, ...]
    Output: (x_center, y_center, width, height) — normalized
    """
    xs = coords[0::2]  # Every even index = x
    ys = coords[1::2]  # Every odd index = y
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height


def convert_label_file(input_path, output_path, class_mapping, is_segmentation=False):
    """
    Read a label file, remap class IDs, and optionally convert seg→bbox.
    """
    lines = []
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            old_class_id = int(parts[0])
            
            # Skip if class ID not in mapping
            if old_class_id not in class_mapping:
                print(f"  ⚠ Unknown class ID {old_class_id} in {input_path.name}, skipping line")
                continue
            
            new_class_id = class_mapping[old_class_id]
            
            if is_segmentation:
                # Convert polygon to bounding box
                coords = [float(x) for x in parts[1:]]
                if len(coords) < 4:
                    continue
                x_center, y_center, width, height = seg_to_bbox(coords)
                lines.append(f"{new_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            else:
                # Already bounding box format, just remap class
                bbox = " ".join(parts[1:])
                lines.append(f"{new_class_id} {bbox}")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    return len(lines)


def copy_dataset(src_images, src_labels, dst_images, dst_labels, class_mapping, 
                 is_segmentation=False, prefix=""):
    """Copy images and convert labels from source to destination."""
    
    if not src_images.exists():
        print(f"  ⚠ Source not found: {src_images}")
        return 0, 0
    
    image_count = 0
    label_count = 0
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    for img_file in sorted(src_images.iterdir()):
        if img_file.suffix.lower() not in image_extensions:
            continue
        
        # Add prefix to avoid filename collisions
        new_name = f"{prefix}{img_file.name}" if prefix else img_file.name
        
        # Copy image
        dst_img = dst_images / new_name
        shutil.copy2(img_file, dst_img)
        image_count += 1
        
        # Find and convert corresponding label
        label_name = img_file.stem + ".txt"
        src_label = src_labels / label_name
        dst_label = dst_labels / (f"{prefix}{img_file.stem}.txt" if prefix else label_name)
        
        if src_label.exists():
            count = convert_label_file(src_label, dst_label, class_mapping, is_segmentation)
            label_count += 1
        else:
            # Create empty label file (no detections in this image)
            dst_label.touch()
            label_count += 1
    
    return image_count, label_count


def main():
    print("=" * 60)
    print("🔬 DATASET MERGER — Spore Detection")
    print("=" * 60)
    
    # ─── Create merged directory structure ───
    for split in ["train", "val", "test"]:
        (MERGED_DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DATASET / split / "labels").mkdir(parents=True, exist_ok=True)
    
    print("\n📂 Created merged dataset structure")
    
    total_images = 0
    total_labels = 0
    
    # ─── Copy EXISTING dataset (already bbox format) ───
    print("\n📦 Copying EXISTING dataset (magnaporthe_oryzae)...")
    
    # Existing: train
    imgs, lbls = copy_dataset(
        EXISTING_DATASET / "train" / "images",
        EXISTING_DATASET / "train" / "labels",
        MERGED_DATASET / "train" / "images",
        MERGED_DATASET / "train" / "labels",
        EXISTING_TO_MERGED,
        is_segmentation=False,
        prefix="existing_"
    )
    print(f"  ✅ Train: {imgs} images, {lbls} labels")
    total_images += imgs
    total_labels += lbls
    
    # Existing: val
    imgs, lbls = copy_dataset(
        EXISTING_DATASET / "val" / "images",
        EXISTING_DATASET / "val" / "labels",
        MERGED_DATASET / "val" / "images",
        MERGED_DATASET / "val" / "labels",
        EXISTING_TO_MERGED,
        is_segmentation=False,
        prefix="existing_"
    )
    print(f"  ✅ Val:   {imgs} images, {lbls} labels")
    total_images += imgs
    total_labels += lbls
    
    # Existing: test
    imgs, lbls = copy_dataset(
        EXISTING_DATASET / "test" / "images",
        EXISTING_DATASET / "test" / "labels",
        MERGED_DATASET / "test" / "images",
        MERGED_DATASET / "test" / "labels",
        EXISTING_TO_MERGED,
        is_segmentation=False,
        prefix="existing_"
    )
    print(f"  ✅ Test:  {imgs} images, {lbls} labels")
    total_images += imgs
    total_labels += lbls
    
    # ─── Copy NEW dataset (segmentation → bbox conversion) ───
    print("\n📦 Copying NEW dataset (Curvularia genus, converting seg→bbox)...")
    
    # New: train
    imgs, lbls = copy_dataset(
        NEW_DATASET / "train" / "images",
        NEW_DATASET / "train" / "labels",
        MERGED_DATASET / "train" / "images",
        MERGED_DATASET / "train" / "labels",
        NEW_TO_MERGED,
        is_segmentation=True,
        prefix="curv_"
    )
    print(f"  ✅ Train: {imgs} images, {lbls} labels")
    total_images += imgs
    total_labels += lbls
    
    # New: valid → val
    imgs, lbls = copy_dataset(
        NEW_DATASET / "valid" / "images",
        NEW_DATASET / "valid" / "labels",
        MERGED_DATASET / "val" / "images",
        MERGED_DATASET / "val" / "labels",
        NEW_TO_MERGED,
        is_segmentation=True,
        prefix="curv_"
    )
    print(f"  ✅ Val:   {imgs} images, {lbls} labels")
    total_images += imgs
    total_labels += lbls
    
    # New: test
    imgs, lbls = copy_dataset(
        NEW_DATASET / "test" / "images",
        NEW_DATASET / "test" / "labels",
        MERGED_DATASET / "test" / "images",
        MERGED_DATASET / "test" / "labels",
        NEW_TO_MERGED,
        is_segmentation=True,
        prefix="curv_"
    )
    print(f"  ✅ Test:  {imgs} images, {lbls} labels")
    total_images += imgs
    total_labels += lbls
    
    # ─── Create new data.yaml ───
    data_yaml = {
        'path': str(MERGED_DATASET.resolve()).replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(MERGED_CLASSES),
        'names': MERGED_CLASSES,
    }
    
    yaml_path = PROJECT_ROOT / "configs" / "data_merged.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n📄 Created config: {yaml_path}")
    
    # ─── Summary ───
    print("\n" + "=" * 60)
    print("✅ MERGE COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Total: {total_images} images, {total_labels} labels")
    print(f"\n🏷️  Classes ({len(MERGED_CLASSES)}):")
    for i, name in enumerate(MERGED_CLASSES):
        marker = " ⭐ TARGET" if i == 0 else ""
        print(f"  {i}: {name}{marker}")
    
    print(f"\n🚀 To train, run:")
    print(f'   python scripts/train.py --data configs/data_merged.yaml --epochs 100')
    print()


if __name__ == "__main__":
    main()
