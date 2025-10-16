# Exercise Recognition using Pose Estimation

A deep learning-based system for real-time exercise recognition using MediaPipe pose estimation and Vision Transformer (ViT) architecture.

## 🎯 Project Overview

This project implements an end-to-end pipeline for recognizing different exercise types from video data. It uses MediaPipe for pose landmark extraction and trains a Vision Transformer model for temporal sequence classification.

### Key Features
- **Multi-Model Support**: LSTM and Vision Transformer (ViT) architectures
- **Real-time Inference**: Webcam application with live predictions
- **Robust Pre-processing**: MediaPipe-based pose extraction with normalization
- **Data Augmentation**: Temporal jitter, spatial noise, random temporal crop
- **Comprehensive Evaluation**: Confusion matrix, classification reports, metrics
- **Exercise Classes**: BenchPress, BodyWeightSquats, JumpRope, Lunges, Pushups, idle

---

## 🏗️ Model Architecture

### Primary Model: Vision Transformer (ViT)

**Architecture Details:**
```
Input: (batch_size, sequence_length=20, input_dim=99)
  ↓
Patch Embedding: Linear(99 → 128)
  ↓
Positional Encoding: Sinusoidal encoding
  ↓
Class Token: Prepended learnable token
  ↓
Transformer Encoder: 4 layers
  - Multi-Head Attention (4 heads)
  - Feed-Forward Network (MLP ratio=4)
  - Pre-norm LayerNorm
  - Dropout=0.1
  ↓
Classification Head: Linear(128 → num_classes)
```

**Model Parameters:**
- Embedding Dimension: 128
- Depth: 4 transformer layers
- Attention Heads: 4
- MLP Ratio: 4
- Dropout: 0.1
- Temporal Window: 20 frames
- Total Parameters: ~0.6M (depends on num_classes)

**Why Vision Transformer?**
1. **Temporal Modeling**: Captures long-range dependencies in pose sequences
2. **Parallel Processing**: More efficient training on GPUs compared to sequential LSTM
3. **Attention Mechanism**: Learns which frames are most important for classification
4. **Memory Efficient**: Optimized for 4GB VRAM with smaller model size

**Alternative Models Available:**
- **LSTM**: Bidirectional LSTM with 2 layers (128 hidden dim)

---

## 🔧 Pre-processing Pipeline

### 1. Pose Extraction
**Tool**: MediaPipe Pose (Complexity=1)
```
Video → MediaPipe → 33 Keypoints × 3 Coordinates = 99 features/frame
```

**MediaPipe Landmarks:**
- 33 body keypoints (face, torso, arms, legs)
- Each landmark: (x, y, visibility)
- Normalized coordinates [0, 1]

### 2. Normalization
**Hip-Centered Normalization:**
```python
1. Center: Subtract hip midpoint (landmarks 23, 24)
2. Scale: Divide by torso height (hip-to-shoulder distance)
3. Result: Translation and scale-invariant poses
```

**Benefits:**
- Person-agnostic and camera-angle invariant
- Focuses on body movement patterns rather than appearance

### 3. Temporal Windowing
- **Sequence Length**: 20 frames (~0.7 seconds at 30fps)
- **Stride**: Variable based on augmentation settings
- **Optimized for 4GB VRAM**: Shorter sequences for memory efficiency

### 4. Data Augmentation (Training Only)
1. **Temporal Jitter**: Random temporal cropping and stretching
2. **Spatial Noise**: Add Gaussian noise (σ=0.02) to coordinates
3. **Random Temporal Crop**: Randomly select subsequences for training

---

## 📊 Training Details

### Hyperparameters
```yaml
Epochs: 30
Batch Size: 48
Learning Rate: 0.0003
Weight Decay: 0.0001
Optimizer: AdamW
Scheduler: Cosine Annealing
Warmup Epochs: 3
Early Stopping Patience: 7
GPU Optimized: GTX 1650 (4GB VRAM)
```

### Training Process
1. **Data Loading**:
   - Windows: num_workers=0, pin_memory=True
   - GPU-optimized for GTX 1650 (4GB VRAM)
   - Smaller model size for memory efficiency

2. **Loss Function**: Cross-Entropy Loss

3. **Monitoring**:
   - TensorBoard logging
   - Validation after each epoch
   - Best model checkpoint saving

4. **Early Stopping**:
   - Stops if validation accuracy doesn't improve for 7 epochs
   - Minimum delta: 0.001
   - Saves best model automatically

### Dataset Split
```
Train: 70% (with augmentation)
Validation: 15%
Test: 15%
Seed: 42 (for reproducibility)
```

---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/GeoJoseph057/Exercise_Recognition.git
cd exercise_recognition
```

### 2. Create Environment
```bash
# Using venv
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Test MediaPipe
python src/preprocessing/pose_extractor.py

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📁 Project Structure

```
exercise_recognition/
├── configs/
│   └── config.yaml              # Training configuration
├── data/
│   ├── raw/                     # Original videos (organized by class)
│   │   ├── BenchPress/
│   │   ├── BodyWeightSquats/
│   │   ├── JumpRope/
│   │   ├── Lunges/
│   │   ├── Pushups/
│   │   └── idle/
│   ├── processed/               # Extracted poses (.npz + .json files)
│   │   ├── classes.json         # Class labels
│   │   └── splits/              # train/val/test JSON files
│   └── evaluation_results/      # Test results and predictions
├── models/
│   ├── checkpoints/             # Training checkpoints
│   ├── final/                   # Final trained models
│   └── onnx/                    # ONNX export models
├── scripts/
│   ├── extract_poses.py         # Extract poses from videos
│   ├── create_splits.py         # Create data splits
│   └── fix_split_paths.py       # Fix split file paths
├── src/
│   ├── models/
│   │   └── pose_encoder.py      # Model architectures
│   ├── preprocessing/
│   │   └── pose_extractor.py    # MediaPipe wrapper
│   ├── training/
│   │   ├── train.py             # Training script
│   │   └── evaluate.py          # Evaluation script
│   ├── inference/
│   │   └── webcam_app.py        # Real-time webcam app
│   └── utils/
│       └── dataset.py           # PyTorch Dataset
├── runs/                        # TensorBoard logs
├── requirements.txt
├── find_checkpoint.py          # Utility to find checkpoints
└── README.md
```

---

## 🎬 Usage Guide

### Step 1: Prepare Dataset

**Organize your videos:**
```
data/raw/
├── BenchPress/
│   └── video1.avi
├── BodyWeightSquats/
│   └── video1.avi
├── JumpRope/
│   └── video1.avi
├── Lunges/
│   └── video1.avi
├── Pushups/
│   ├── video1.avi
│   └── video2.avi
└── idle/
    ├── video1.mp4
```

**Extract poses:**
```bash
python scripts/extract_poses.py --input data/raw --output data/processed
```

**Create splits:**
```bash
python scripts/create_splits.py --data_dir data/processed
```

### Step 2: Train Model

**Start training:**
```bash
python src/training/train.py --config configs/config.yaml
```

**Optional: Monitor training:**
```bash
tensorboard --logdir runs/
```

**Optional: Customize training**
```bash
python src/training/train.py \
    --config configs/config.yaml \
    --model vit \
    --epochs 30 \
    --batch_size 16
```

### Step 3: Evaluate Model

**Find your checkpoint:**
```bash
python find_checkpoint.py
```

**Evaluate on test set:**
```bash
python src/training/evaluate.py \
    --checkpoint models/<checkpoints>/best.pth \
    --split test
```

**Outputs:**
- Confusion matrix (PNG)
- Classification report (TXT)
- Predictions JSON

### Step 4: Run Webcam Inference

**Real-time recognition:**
```bash
python src/inference/webcam_app.py \
    --model models/<checkpoints>/best.pth \
    --config configs/config.yaml
```

**Controls:**
- `Q`: Quit
- `R`: Reset buffer
- `S`: Screenshot

**Optional parameters:**
```bash
python src/inference/webcam_app.py \
    --model models/checkpoints/best.pth \
    --config configs/config.yaml \
    --camera 0 \
    --confidence 0.85 \
    --save_video \
    --output demo.mp4
```

---

## 📈 Results

After training, evaluation results will be saved in `evaluation_results/` directory:
- `test_report.txt`: Detailed classification report
- `test_predictions.json`: Individual predictions for analysis
- `confusion_matrix.png`: Visual confusion matrix

---

## 🔍 Troubleshooting

### Common Issues

**1. MediaPipe not detecting pose**
- Ensure full body is visible in frame
- Check lighting conditions
- Reduce model_complexity in config

**2. CUDA out of memory**
- Reduce batch_size in config (default: 48 for GTX 1650)
- Use smaller model (reduce embed_dim, depth)

**3. Windows multiprocessing error**
- Already fixed: num_workers=0 in dataset.py

**4. Low accuracy**
- Insufficient training data (need 20+ videos per class)
- Class imbalance (balance your dataset)
- Too short videos (need 20+ frames)
- Inconsistent exercise form in videos

---

## 🎓 Model Training Tips

### For Best Results:

1. **Data Quality**
   - Record 50-100 videos per exercise class
   - Ensure full body visibility and consistent lighting
   - Clear exercise execution (3-10 seconds per video)

2. **Training Tips**
   - Start with default config (optimized for GTX 1650)
   - Monitor GPU usage with `nvidia-smi`
   - Increase epochs if validation loss still decreasing

---

## 🔬 Technical Details

### Input Format
```python
Shape: (batch, sequence_length, features)
      = (48, 20, 99)

Where:
- batch: 48 sequences (optimized for GTX 1650)
- sequence_length: 20 frames (~0.7 seconds)
- features: 99 (33 keypoints × 3 coords)
```

### Model Output
```python
Logits: (batch, num_classes)
Probabilities: softmax(logits)
Prediction: argmax(probabilities)
```

### Inference Pipeline
```
Webcam Frame
    ↓
MediaPipe Pose Detection
    ↓
Extract 33 Landmarks (99 features)
    ↓
Normalize (hip-centered, scaled)
    ↓
Add to Temporal Buffer (20 frames)
    ↓
Model Inference (ViT forward pass)
    ↓
Softmax → Class Prediction
    ↓
Temporal Smoothing (5 frames)
    ↓
Display on Screen
```

---
