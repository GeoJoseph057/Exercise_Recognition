"""
PyTorch Dataset for Pose-based Exercise Recognition
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import random


class PoseSequenceDataset(Dataset):
    """
    Dataset for pose sequences
    Each sample is a temporal sequence of pose landmarks
    """

    def __init__(
        self,
        data_dir,
        split='train',
        sequence_length=30,
        stride=15,
        augment=False,
        normalize=True
    ):
        """
        Args:
            data_dir: directory containing processed pose files
            split: 'train', 'val', or 'test'
            sequence_length: number of frames per sequence
            stride: stride for sliding window (only for train)
            augment: whether to apply data augmentation
            normalize: whether to normalize poses
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride if split == 'train' else sequence_length
        self.augment = augment and split == 'train'
        self.normalize = normalize

        # Load split file
        split_file = self.data_dir / 'splits' / f'{split}.json'
        with open(split_file, 'r') as f:
            self.samples = json.load(f)

        # Load class mapping
        class_file = self.data_dir / 'classes.json'
        with open(class_file, 'r') as f:
            self.classes = json.load(f)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Create windowed sequences
        self.sequences = []
        self._create_sequences()

        print(f"{split.upper()} Dataset: {len(self.sequences)} sequences from {len(self.samples)} videos")

    def _create_sequences(self):
        """Create sliding window sequences from videos"""
        for sample in self.samples:
            pose_file = self.data_dir / 'processed' / sample['pose_file']
            label = self.class_to_idx[sample['label']]

            # Load pose data
            poses = np.load(pose_file)
            num_frames = len(poses)

            # Create windows
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                self.sequences.append({
                    'pose_file': pose_file,
                    'start_idx': start_idx,
                    'end_idx': start_idx + self.sequence_length,
                    'label': label,
                    'video_id': sample.get('video_id', pose_file.stem)
                })

    def _normalize_pose(self, pose_seq):
        """
        Normalize pose sequence
        Center around hip and scale by torso height
        """
        # Hip center (landmarks 23, 24)
        left_hip = pose_seq[:, 23, :2]
        right_hip = pose_seq[:, 24, :2]
        hip_center = (left_hip + right_hip) / 2

        # Center
        pose_seq[:, :, :2] -= hip_center[:, np.newaxis, :]

        # Shoulder center (landmarks 11, 12)
        left_shoulder = pose_seq[:, 11, :2]
        right_shoulder = pose_seq[:, 12, :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # Torso height
        torso_height = np.linalg.norm(shoulder_center - hip_center, axis=1, keepdims=True)
        torso_height = np.maximum(torso_height, 0.01)

        # Scale
        pose_seq[:, :, :2] /= torso_height[:, np.newaxis, :]

        return pose_seq

    def _augment_pose(self, pose_seq):
        """
        Data augmentation for pose sequences
        """
        pose_seq = pose_seq.copy()

        # 1. Random temporal crop (slight variation in start/end)
        if random.random() < 0.3:
            crop_frames = random.randint(1, 3)
            if len(pose_seq) > crop_frames:
                start = random.randint(0, crop_frames)
                pose_seq = pose_seq[start:start + self.sequence_length]

        # 2. Horizontal flip (mirror)
        if random.random() < 0.5:
            pose_seq[:, :, 0] = -pose_seq[:, :, 0]  # Flip x coordinate
            # Swap left-right landmarks
            left_right_pairs = [
                (11, 12), (13, 14), (15, 16),  # Arms
                (17, 18), (19, 20), (21, 22),  # Hands
                (23, 24), (25, 26), (27, 28),  # Legs
                (29, 30), (31, 32)  # Feet
            ]
            for left, right in left_right_pairs:
                pose_seq[:, [left, right]] = pose_seq[:, [right, left]]

        # 3. Add Gaussian noise to coordinates
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.02, pose_seq[:, :, :2].shape)
            pose_seq[:, :, :2] += noise

        # 4. Random temporal scaling (speed up/slow down)
        if random.random() < 0.3:
            scale_factor = random.uniform(0.8, 1.2)
            new_length = int(len(pose_seq) * scale_factor)
            if new_length > 0:
                indices = np.linspace(0, len(pose_seq) - 1, self.sequence_length).astype(int)
                pose_seq = pose_seq[indices]

        # 5. Random rotation (around z-axis)
        if random.random() < 0.3:
            angle = random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            xy = pose_seq[:, :, :2]
            pose_seq[:, :, :2] = np.dot(xy, rotation_matrix.T)

        return pose_seq

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns:
            pose_seq: (sequence_length, 33, 3) -> flattened to (sequence_length, 99)
            label: int
        """
        seq_info = self.sequences[idx]

        # Load full pose sequence
        poses = np.load(seq_info['pose_file'])

        # Extract window
        pose_seq = poses[seq_info['start_idx']:seq_info['end_idx']].copy()

        # Ensure correct length (pad if necessary)
        if len(pose_seq) < self.sequence_length:
            padding = np.repeat(pose_seq[-1:], self.sequence_length - len(pose_seq), axis=0)
            pose_seq = np.concatenate([pose_seq, padding], axis=0)

        # Normalize
        if self.normalize:
            pose_seq = self._normalize_pose(pose_seq)

        # Augment
        if self.augment:
            pose_seq = self._augment_pose(pose_seq)

        # Flatten landmarks: (seq_len, 33, 3) -> (seq_len, 99)
        pose_seq = pose_seq.reshape(self.sequence_length, -1)

        # Convert to tensor
        pose_tensor = torch.from_numpy(pose_seq).float()
        label = torch.tensor(seq_info['label'], dtype=torch.long)

        return pose_tensor, label


def create_dataloaders(
    data_dir,
    batch_size=32,
    sequence_length=30,
    num_workers=4,
    pin_memory=True
):
    """
    Create train, val, and test dataloaders

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    # Training dataset with augmentation
    train_dataset = PoseSequenceDataset(
        data_dir=data_dir,
        split='train',
        sequence_length=sequence_length,
        stride=15,  # 50% overlap
        augment=True,
        normalize=True
    )

    # Validation dataset
    val_dataset = PoseSequenceDataset(
        data_dir=data_dir,
        split='val',
        sequence_length=sequence_length,
        stride=sequence_length,  # No overlap
        augment=False,
        normalize=True
    )

    # Test dataset
    test_dataset = PoseSequenceDataset(
        data_dir=data_dir,
        split='test',
        sequence_length=sequence_length,
        stride=sequence_length,
        augment=False,
        normalize=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, train_dataset.num_classes


# Example usage
if __name__ == "__main__":
    # Test dataset
    data_dir = "data"

    # Create dummy data structure for testing
    print("Creating test dataset...")

    dataset = PoseSequenceDataset(
        data_dir=data_dir,
        split='train',
        sequence_length=30,
        augment=True
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Classes: {dataset.classes}")

    # Test batch
    pose_seq, label = dataset[0]
    print(f"\nSample shape: {pose_seq.shape}")  # (30, 99)
    print(f"Label: {label} ({dataset.classes[label]})")

    # Test dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    batch_poses, batch_labels = next(iter(loader))
    print(f"\nBatch poses shape: {batch_poses.shape}")  # (8, 30, 99)
    print(f"Batch labels shape: {batch_labels.shape}")  # (8,)