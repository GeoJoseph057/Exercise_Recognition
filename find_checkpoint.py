"""
Find all model checkpoints in the project
"""

from pathlib import Path
import os

def find_checkpoints():
    """Find all .pth checkpoint files"""

    print("🔍 Searching for checkpoints...\n")

    # Search in models directory
    models_dir = Path('models')

    if not models_dir.exists():
        print("❌ models/ directory doesn't exist!")
        return

    # Find all .pth files
    checkpoint_files = list(models_dir.rglob('*.pth'))

    if not checkpoint_files:
        print("❌ No .pth checkpoint files found!")
        print("\nPossible reasons:")
        print("  1. Training hasn't been run yet")
        print("  2. Training failed to save checkpoints")
        print("  3. Checkpoints are in a different location")
        return

    print(f"✅ Found {len(checkpoint_files)} checkpoint file(s):\n")

    # Get current working directory as absolute path
    cwd = Path.cwd()

    for i, ckpt in enumerate(checkpoint_files, 1):
        # Resolve to absolute path before getting relative path
        ckpt_abs = ckpt.resolve()

        # Get file size
        size_mb = ckpt_abs.stat().st_size / (1024 * 1024)

        # Get relative path from project root
        rel_path = ckpt_abs.relative_to(cwd)

        print(f"{i}. {rel_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Full path: {ckpt_abs}")

        # Check parent directory for metadata
        parent = ckpt_abs.parent
        config_file = parent / 'config.yaml'
        history_file = parent / 'history.json'

        if config_file.exists():
            print(f"   ✓ Has config.yaml")
        if history_file.exists():
            print(f"   ✓ Has history.json")

        print()

    # Suggest the most recent one
    latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    latest_abs = latest.resolve()
    latest_rel = latest_abs.relative_to(cwd)

    print("="*60)
    print(f"📌 Most recent checkpoint: {latest_rel}")
    print("="*60)

    # Generate command
    print("\n💡 To test with webcam, run:")
    print(f"python src/inference/webcam_app.py --model {latest_rel} --config configs/config.yaml")

    print("\n💡 To evaluate, run:")
    print(f"python src/training/evaluate.py --checkpoint {latest_rel} --split test")

def check_directory_structure():
    """Check if directories exist"""
    print("\n📁 Checking directory structure:")

    dirs_to_check = [
        'models',
        'models/checkpoints',
        'models/final',
        'configs',
        'data/processed'
    ]

    for dir_path in dirs_to_check:
        path = Path(dir_path)
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {dir_path}")

if __name__ == '__main__':
    find_checkpoints()
    check_directory_structure()
