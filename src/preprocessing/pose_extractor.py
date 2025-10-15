"""
MediaPipe Pose Extraction Utility
Extracts pose landmarks from videos for exercise recognition
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import json
from mediapipe.framework.formats import landmark_pb2


class PoseExtractor:
    """Extract pose landmarks from videos using MediaPipe"""

    def __init__(
        self,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        enable_segmentation=False
    ):
        """
        Initialize MediaPipe Pose

        Args:
            model_complexity: 0, 1, or 2 (higher = more accurate but slower)
            min_detection_confidence: minimum confidence for person detection
            min_tracking_confidence: minimum confidence for pose tracking
            static_image_mode: if True, treats each frame independently
            enable_segmentation: segmentation mask (disable for better performance)
        """
        # Force MediaPipe to use CPU to avoid GPU conflicts
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU for MediaPipe

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
            enable_segmentation=enable_segmentation  # Disable for performance
        )

        # MediaPipe returns 33 landmarks
        self.num_landmarks = 33

    def extract_from_frame(self, frame):
        """
        Extract pose from a single frame

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            landmarks: np.array of shape (33, 3) or None if no pose detected
                      Each landmark has (x, y, visibility)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([
                    landmark.x,  # Normalized [0, 1]
                    landmark.y,  # Normalized [0, 1]
                    landmark.visibility  # [0, 1]
                ])
            return np.array(landmarks, dtype=np.float32)

        return None

    def extract_from_video(self, video_path, max_frames=None, target_fps=None):
        """
        Extract pose sequence from video

        Args:
            video_path: path to video file
            max_frames: maximum number of frames to process (None = all)
            target_fps: resample video to this fps (None = use original)

        Returns:
            pose_sequence: list of pose landmarks, each of shape (33, 3)
            metadata: dict with video info
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame sampling
        if target_fps and target_fps < fps:
            frame_step = int(fps / target_fps)
        else:
            frame_step = 1
            target_fps = fps

        pose_sequence = []
        frame_indices = []
        frame_count = 0
        frames_processed = 0

        with tqdm(total=min(max_frames or total_frames, total_frames),
                  desc=f"Extracting poses from {Path(video_path).name}") as pbar:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames
                if frame_count % frame_step == 0:
                    # Extract pose
                    landmarks = self.extract_from_frame(frame)

                    if landmarks is not None:
                        pose_sequence.append(landmarks)
                        frame_indices.append(frame_count)

                    frames_processed += 1
                    pbar.update(1)

                    if max_frames and frames_processed >= max_frames:
                        break

                frame_count += 1

        cap.release()

        metadata = {
            'video_path': str(video_path),
            'original_fps': fps,
            'target_fps': target_fps,
            'total_frames': total_frames,
            'frames_extracted': len(pose_sequence),
            'frame_indices': frame_indices,
            'resolution': (width, height)
        }

        return pose_sequence, metadata

    def visualize_pose(self, frame, landmarks):
        """
        Draw pose landmarks on frame

        Args:
            frame: BGR image
            landmarks: pose landmarks array (33, 3)

        Returns:
            annotated_frame: frame with pose drawn
        """
        # Convert landmarks back to MediaPipe format
        pose_landmarks = self.mp_pose.PoseLandmark
        landmark_list = []

        for i, lm in enumerate(landmarks):
            landmark_list.append(
                landmark_pb2.NormalizedLandmark(
                    x=lm[0], y=lm[1], visibility=lm[2]
                )
            )

        # Create pose landmarks object
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList(landmark=landmark_list)

        # Draw
        annotated_frame = frame.copy()
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            pose_landmarks_proto,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        return annotated_frame

    def save_poses(self, pose_sequence, metadata, output_path):
        """
        Save pose sequence to file

        Args:
            pose_sequence: list of pose landmarks
            metadata: metadata dict
            output_path: path to save (.npz or .npy)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.npz':
            # Save as compressed npz with metadata
            np.savez_compressed(
                output_path,
                poses=np.array(pose_sequence),
                **metadata
            )
        elif output_path.suffix == '.npy':
            # Save just poses
            np.save(output_path, np.array(pose_sequence))
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}")

        # Save metadata separately as JSON
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            # Convert non-serializable types
            meta_copy = metadata.copy()
            meta_copy['frame_indices'] = [int(x) for x in meta_copy.get('frame_indices', [])]
            json.dump(meta_copy, f, indent=2)

    def load_poses(self, input_path):
        """
        Load pose sequence from file

        Args:
            input_path: path to .npz or .npy file

        Returns:
            pose_sequence: numpy array of poses
            metadata: dict (if available)
        """
        input_path = Path(input_path)

        if input_path.suffix == '.npz':
            data = np.load(input_path, allow_pickle=True)
            pose_sequence = data['poses']
            metadata = {k: data[k].item() if data[k].ndim == 0 else data[k].tolist()
                       for k in data.files if k != 'poses'}
            return pose_sequence, metadata
        elif input_path.suffix == '.npy':
            pose_sequence = np.load(input_path)
            # Try to load metadata
            metadata_path = input_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            return pose_sequence, metadata
        else:
            raise ValueError(f"Unsupported format: {input_path.suffix}")

    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()


def normalize_pose_sequence(pose_sequence):
    """
    Normalize pose sequence for better learning

    Args:
        pose_sequence: np.array of shape (num_frames, 33, 3)

    Returns:
        normalized_sequence: normalized poses
    """
    sequence = np.array(pose_sequence).copy()

    # Method 1: Center around hip midpoint
    # Hip landmarks: 23 (left), 24 (right)
    left_hip = sequence[:, 23, :2]  # (num_frames, 2)
    right_hip = sequence[:, 24, :2]
    hip_center = (left_hip + right_hip) / 2  # (num_frames, 2)

    # Subtract hip center from x, y coordinates
    sequence[:, :, :2] -= hip_center[:, np.newaxis, :]

    # Method 2: Scale by torso height
    # Shoulders: 11 (left), 12 (right)
    left_shoulder = sequence[:, 11, :2]
    right_shoulder = sequence[:, 12, :2]
    shoulder_center = (left_shoulder + right_shoulder) / 2

    # Calculate torso height (shoulder to hip distance)
    torso_height = np.linalg.norm(shoulder_center - hip_center, axis=1, keepdims=True)
    torso_height = np.maximum(torso_height, 0.01)  # Avoid division by zero

    # Scale coordinates
    sequence[:, :, :2] /= torso_height[:, np.newaxis, :]

    return sequence


# Example usage
if __name__ == "__main__":
    # Test with webcam or video
    print("Initializing MediaPipe Pose (CPU mode for stability)...")
    extractor = PoseExtractor(
        model_complexity=0,  # Use 0 for fastest performance
        enable_segmentation=False
    )

    # Test with webcam (press 'q' to quit)
    print("\n🎥 Testing with webcam...")
    print("Controls:")
    print("  - Press 'Q' to quit")
    print("  - Press 'D' to toggle drawing")
    print("  - Keep your full body in frame for best results\n")

    cap = cv2.VideoCapture(0)

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        print("Troubleshooting:")
        print("  1. Check if camera is connected")
        print("  2. Try different camera_id: cv2.VideoCapture(1)")
        print("  3. Close other apps using the camera")
        sys.exit(1)

    print(f"✅ Camera opened successfully")
    print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"   FPS: {int(cap.get(cv2.CAP_PROP_FPS))}\n")

    draw_pose = True
    frame_count = 0
    pose_detected_count = 0
    detection_rate = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            frame_count += 1

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)

            # Extract pose
            landmarks = extractor.extract_from_frame(frame)

            if landmarks is not None:
                pose_detected_count += 1

                # Visualize
                if draw_pose:
                    annotated = extractor.visualize_pose(frame, landmarks)

                    # Add info text
                    cv2.putText(annotated, "Pose Detected!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated, f"Landmarks: {len(landmarks)}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    frame = annotated
                else:
                    cv2.putText(frame, "Pose Detected (Drawing OFF)", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Pose Detected", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Move into frame & show full body", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add FPS and stats
            detection_rate = (pose_detected_count / frame_count) * 100 if frame_count > 0 else 0
            cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'Q' to quit | 'D' to toggle drawing", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Show frame
            cv2.imshow('MediaPipe Pose Test', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('d') or key == ord('D'):
                draw_pose = not draw_pose
                print(f"Drawing {'enabled' if draw_pose else 'disabled'}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Camera released")

    print("\n📊 Session Statistics:")
    print(f"   Total frames: {frame_count}")
    print(f"   Poses detected: {pose_detected_count}")
    detection_rate = (pose_detected_count / frame_count) * 100 if frame_count > 0 else 0
    print(f"   Detection rate: {detection_rate:.1f}%")
    print("\nPose extractor ready!")
    print(f"Number of landmarks: {extractor.num_landmarks}")
    print(f"Each landmark has: x, y, visibility")
    print(f"Total features per frame: {extractor.num_landmarks * 3}")