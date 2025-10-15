"""
Fix paths in split JSON files
"""

import json
from pathlib import Path

def fix_split_file(split_path):
    """Fix paths in a split JSON file"""
    with open(split_path, 'r') as f:
        data = json.load(f)

    fixed_data = []
    for item in data:
        pose_file = item['pose_file']

        # Remove 'processed/' prefix if it exists
        if pose_file.startswith('processed/'):
            pose_file = pose_file.replace('processed/', '', 1)

        # Update item
        item['pose_file'] = pose_file
        fixed_data.append(item)

    # Save fixed data
    with open(split_path, 'w') as f:
        json.dump(fixed_data, f, indent=2)

    print(f"✅ Fixed {split_path}")
    print(f"   Example path: {fixed_data[0]['pose_file']}")

if __name__ == '__main__':
    data_dir = Path('data')
    splits_dir = data_dir / 'splits'

    if not splits_dir.exists():
        print(f"❌ Splits directory not found: {splits_dir}")
        exit(1)

    for split_file in ['train.json', 'val.json', 'test.json']:
        split_path = splits_dir / split_file
        if split_path.exists():
            fix_split_file(split_path)
        else:
            print(f"⚠️ Split file not found: {split_path}")