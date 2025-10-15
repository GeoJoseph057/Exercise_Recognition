"""
Extract poses from all videos in dataset
Usage: python scripts/extract_poses.py --input data/raw --output data/processed
"""

import argparse
from pathlib import Path
import sys
sys.path.append('src')
from preprocessing.pose_extractor import PoseExtractor
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--target_fps', type=int, default=30)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pose extractor
    extractor = PoseExtractor(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Find all videos
    video_extensions = ['.mp4', '.avi', '.mov', '.MOV', '.MP4', '.AVI']
    video_files = []

    for ext in video_extensions:
        video_files.extend(input_dir.rglob(f'*{ext}'))

    print(f"Found {len(video_files)} videos")

    # Process each video
    failed_videos = []
    for video_path in tqdm(video_files, desc="Extracting poses"):
        try:
            # Get relative path to maintain folder structure
            rel_path = video_path.relative_to(input_dir)

            # Create output path
            output_path = output_dir / rel_path.with_suffix('.npz')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if already processed
            if output_path.exists():
                continue

            # Extract poses
            pose_seq, metadata = extractor.extract_from_video(
                video_path,
                target_fps=args.target_fps
            )

            if len(pose_seq) == 0:
                print(f"⚠️ No poses detected in {video_path.name}")
                failed_videos.append(str(video_path))
                continue

            # Save
            extractor.save_poses(pose_seq, metadata, output_path)

        except Exception as e:
            print(f"❌ Error processing {video_path.name}: {e}")
            failed_videos.append(str(video_path))

    print(f"\n✅ Processing complete!")
    print(f"Failed videos: {len(failed_videos)}")
    if failed_videos:
        print("Failed files:")
        for f in failed_videos:
            print(f"  - {f}")

if __name__ == '__main__':
    main()