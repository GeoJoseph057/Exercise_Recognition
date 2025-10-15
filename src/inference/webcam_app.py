"""
Real-time Exercise Recognition using Webcam
Uses trained model + MediaPipe for live inference
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from pathlib import Path
import argparse
import json
import yaml
from collections import deque
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.pose_encoder import get_model


class ExerciseRecognizer:
    """Real-time exercise recognition system"""

    def __init__(
        self,
        model_path,
        config_path=None,
        sequence_length=30,
        confidence_threshold=0.6,
        smoothing_window=5
    ):
        """
        Args:
            model_path: path to trained model checkpoint
            config_path: path to config.yaml (optional)
            sequence_length: number of frames for temporal window
            confidence_threshold: minimum confidence for prediction
            smoothing_window: number of predictions to smooth over
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window

        # Load model
        self._load_model(model_path, config_path)

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )

        # Temporal buffer for pose sequences
        self.pose_buffer = deque(maxlen=sequence_length)

        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=smoothing_window)

        # FPS tracking
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()

    def _load_model(self, model_path, config_path):
        """Load trained model"""
        print(f"Loading model from {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Get config
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("Config not found. Please provide config_path")

        self.config = config

        # Load classes
        data_dir = Path(config['data']['processed_path'])
        classes_file = data_dir / 'classes.json'
        with open(classes_file, 'r') as f:
            self.classes = json.load(f)

        print(f"Classes: {self.classes}")

        # Create model
        model_type = config['model']['type']
        if model_type == 'vit':
            model_config = config['model']['vit']
        elif model_type == 'lstm':
            model_config = config['model']['lstm']
        else:
            model_config = {}

        self.model = get_model(
            model_type=model_type,
            num_classes=len(self.classes),
            config=model_config
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✅ Model loaded successfully!")
        print(f"   Best Val Acc: {checkpoint.get('val_acc', 'N/A')}")

    def extract_pose(self, frame):
        """Extract pose landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.visibility])
            return np.array(landmarks, dtype=np.float32)

        return None

    def normalize_pose(self, pose):
        """Normalize pose (same as training)"""
        # Hip center
        left_hip = pose[23, :2]
        right_hip = pose[24, :2]
        hip_center = (left_hip + right_hip) / 2

        # Center
        pose[:, :2] -= hip_center

        # Shoulder center
        left_shoulder = pose[11, :2]
        right_shoulder = pose[12, :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # Torso height
        torso_height = np.linalg.norm(shoulder_center - hip_center)
        torso_height = max(torso_height, 0.01)

        # Scale
        pose[:, :2] /= torso_height

        return pose

    def predict(self):
        """Make prediction from current pose buffer"""
        if len(self.pose_buffer) < self.sequence_length:
            return None, 0.0

        # Prepare input
        pose_seq = np.array(list(self.pose_buffer))
        pose_seq = pose_seq.reshape(self.sequence_length, -1)  # (seq_len, 99)
        pose_tensor = torch.from_numpy(pose_seq).float().unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(pose_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(1)

            pred_class = self.classes[pred_idx.item()]
            confidence = confidence.item()

        # Add to smoothing buffer
        self.prediction_buffer.append((pred_class, confidence))

        # Smooth predictions
        if len(self.prediction_buffer) >= self.smoothing_window:
            # Vote for most common prediction with high confidence
            valid_preds = [(c, conf) for c, conf in self.prediction_buffer
                          if conf >= self.confidence_threshold]

            if valid_preds:
                # Weighted voting
                class_votes = {}
                for c, conf in valid_preds:
                    class_votes[c] = class_votes.get(c, 0) + conf

                smoothed_class = max(class_votes, key=class_votes.get)
                smoothed_conf = class_votes[smoothed_class] / len(valid_preds)

                return smoothed_class, smoothed_conf

        return pred_class, confidence

    def draw_info(self, frame, pred_class, confidence):
        """Draw prediction and info on frame"""
        h, w = frame.shape[:2]

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_buffer.append(fps)
        avg_fps = np.mean(self.fps_buffer)

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Draw prediction
        if pred_class:
            # Color based on confidence
            if confidence >= 0.8:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange

            text = f"Exercise: {pred_class.upper()}"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1.2, color, 3)

            conf_text = f"Confidence: {confidence:.1%}"
            cv2.putText(frame, conf_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No prediction", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Draw FPS
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        # Draw buffer status
        buffer_status = f"Buffer: {len(self.pose_buffer)}/{self.sequence_length}"
        cv2.putText(frame, buffer_status, (w - 250, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Instructions
        instructions = [
            "Press 'Q' to quit",
            "Press 'R' to reset buffer",
            "Press 'S' to screenshot"
        ]
        y_offset = h - 100
        for i, inst in enumerate(instructions):
            cv2.putText(frame, inst, (20, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def run(self, camera_id=0, save_video=False, output_path='output.mp4'):
        """Run real-time recognition"""
        print("\n🎥 Starting webcam...")
        print("Press 'Q' to quit, 'R' to reset, 'S' for screenshot")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera: {width}x{height} @ {fps}fps")

        # Video writer (optional)
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        screenshot_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Extract pose
                landmarks = self.extract_pose(frame)

                if landmarks is not None:
                    # Normalize
                    normalized_pose = self.normalize_pose(landmarks.copy())

                    # Add to buffer
                    self.pose_buffer.append(normalized_pose)

                    # Draw pose on frame
                    self.mp_drawing.draw_landmarks(
                        frame,
                        self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )

                    # Predict
                    pred_class, confidence = self.predict()
                else:
                    pred_class, confidence = None, 0.0

                # Draw info
                frame = self.draw_info(frame, pred_class, confidence)

                # Show frame
                cv2.imshow('Exercise Recognition', frame)

                # Save video
                if writer:
                    writer.write(frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    self.pose_buffer.clear()
                    self.prediction_buffer.clear()
                    print("Buffer reset")
                elif key == ord('s') or key == ord('S'):
                    screenshot_path = f'screenshot_{screenshot_count:03d}.jpg'
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved: {screenshot_path}")
                    screenshot_count += 1

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.pose.close()
            print("\n✅ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description='Real-time Exercise Recognition')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml (optional)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--sequence_length', type=int, default=30,
                       help='Temporal window size')
    parser.add_argument('--confidence', type=float, default=0.6,
                       help='Confidence threshold')
    parser.add_argument('--save_video', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='Output video path')
    args = parser.parse_args()

    # Initialize recognizer
    recognizer = ExerciseRecognizer(
        model_path=args.model,
        config_path=args.config,
        sequence_length=args.sequence_length,
        confidence_threshold=args.confidence
    )

    # Run
    recognizer.run(
        camera_id=args.camera,
        save_video=args.save_video,
        output_path=args.output
    )


if __name__ == '__main__':
    main()