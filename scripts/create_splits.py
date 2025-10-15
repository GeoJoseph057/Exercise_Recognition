"""
Create train/val/test splits from processed poses
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

    # Split data
    np.random.seed(args.seed)

    # First split: train + val vs test
    train_val, test = train_test_split(
        samples,
        test_size=args.test_ratio,
        stratify=[s['label'] for s in samples],
        random_state=args.seed
    )

    # Second split: train vs val
    val_ratio_adjusted = args.val_ratio / (args.train_ratio + args.val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=[s['label'] for s in train_val],
        random_state=args.seed
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")

    # Save splits
    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        split_file = splits_dir / f'{split_name}.json'
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"✅ Saved {split_file}")

    # Save class mapping
    classes_file = data_dir / 'classes.json'
    with open(classes_file, 'w') as f:
        json.dump(classes, f, indent=2)
    print(f"✅ Saved {classes_file}")

    print("\n✅ Splits created successfully!")

if __name__ == '__main__':
    main()