"""
Create train/val/test splits from processed poses
Handles tiny datasets by disabling stratification when necessary
Usage: python scripts/create_splits.py --data_dir data/processed
"""

import argparse
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits_dir = data_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)

    # Find all pose files
    pose_files = list(data_dir.rglob('*.npz'))
    print(f"Found {len(pose_files)} pose files")

    if len(pose_files) == 0:
        print("❌ No pose files found! Make sure you've run extract_poses.py first")
        return

    # Extract class labels from folder structure
    samples = []
    classes = set()

    for pose_file in pose_files:
        # Assume structure: data/processed/class_name/video.npz
        class_name = pose_file.parent.name
        classes.add(class_name)

        # Get relative path
        rel_path = pose_file.relative_to(data_dir)

        samples.append({
            'pose_file': str(rel_path),
            'label': class_name,
            'video_id': pose_file.stem
        })

    classes = sorted(list(classes))
    print(f"Classes: {classes}")
    print(f"Number of classes: {len(classes)}")

    # Count samples per class
    class_counts = {c: sum(1 for s in samples if s['label'] == c) for c in classes}
    print("\nSamples per class:")
    for c, count in class_counts.items():
        print(f"  {c}: {count}")

    # Check if dataset is severely imbalanced or too small
    min_samples = min(class_counts.values())
    max_samples = max(class_counts.values())

    # Detect and warn about imbalanced dataset
    if max_samples / min_samples > 10:
        print(f"\n⚠️  WARNING: Severe class imbalance detected!")
        print(f"   Largest class has {max_samples} samples")
        print(f"   Smallest class has {min_samples} samples")
        print(f"   Consider collecting more data for underrepresented classes")

    # Simple random split (no stratification for tiny/imbalanced datasets)
    print(f"\n📊 Using simple random split (stratification disabled for imbalanced data)")

    np.random.seed(args.seed)
    np.random.shuffle(samples)

    # Calculate split indices
    n_total = len(samples)
    n_test = max(1, int(n_total * args.test_ratio))
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_test - n_val

    # Ensure we have at least 1 sample in each split
    if n_train < 1:
        print("❌ Not enough samples to create all splits!")
        print("   You need at least 3 samples total")
        return

    # Split the data
    test = samples[:n_test]
    val = samples[n_test:n_test + n_val]
    train = samples[n_test + n_val:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train)} ({len(train)/n_total*100:.1f}%)")
    print(f"  Val: {len(val)} ({len(val)/n_total*100:.1f}%)")
    print(f"  Test: {len(test)} ({len(test)/n_total*100:.1f}%)")

    # Verify splits - show class distribution
    print("\nClass distribution per split:")
    all_classes = set(classes)

    for split_name, split_data in [('Train', train), ('Val', val), ('Test', test)]:
        split_classes = {}
        for s in split_data:
            split_classes[s['label']] = split_classes.get(s['label'], 0) + 1

        print(f"\n  {split_name}:")
        for cls in classes:
            count = split_classes.get(cls, 0)
            print(f"    {cls}: {count}")

        # Check for missing classes
        missing = all_classes - set(split_classes.keys())
        if missing:
            print(f"    ⚠️  Missing classes: {missing}")

    # Save splits
    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        split_file = splits_dir / f'{split_name}.json'
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"\n✅ Saved {split_file}")

    # Save class mapping
    classes_file = data_dir / 'classes.json'
    with open(classes_file, 'w') as f:
        json.dump(classes, f, indent=2)
    print(f"✅ Saved {classes_file}")

    # Final warnings
    print("\n" + "="*60)
    if min_samples < 10:
        print("⚠️  CRITICAL: Your dataset is too small for real training!")
        print("   Current status:")
        print(f"   - Smallest class: {min_samples} samples (need 20+ per class)")
        print(f"   - Total samples: {n_total}")
        print("\n   Recommendations:")
        print("   1. Collect more videos (aim for 20-50 per class)")
        print("   2. Or use data augmentation to increase samples")
        print("   3. This will only work for testing the pipeline, not real training")
    elif max_samples / min_samples > 5:
        print("⚠️  WARNING: Severe class imbalance detected!")
        print("   This will hurt model performance")
        print("   Consider balancing your dataset or using class weights")
    else:
        print("✅ Splits created successfully!")
    print("="*60)

if __name__ == '__main__':
    main()