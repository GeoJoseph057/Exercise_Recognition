"""
Evaluation script for trained models
Generates confusion matrix, classification report, and metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pathlib import Path
import argparse
import yaml
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.pose_encoder import get_model
from utils.dataset import create_dataloaders


def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and return predictions"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for poses, labels in dataloader:
            poses, labels = poses.to(device), labels.to(device)

            outputs = model(poses)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    args = parser.parse_args()

    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data - FIX: Set num_workers=0 for Windows
    print("Loading data...")
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        data_dir=config['data']['processed_path'],
        batch_size=32,
        sequence_length=config['pose']['temporal_window'],
        num_workers=0,  # FIXED: Set to 0 for Windows
        pin_memory=False  # FIXED: Disable pin_memory when num_workers=0
    )

    # Select dataloader
    if args.split == 'train':
        dataloader = train_loader
    elif args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader

    # Load class names
    data_dir = Path(config['data']['processed_path'])
    with open(data_dir / 'classes.json', 'r') as f:
        class_names = json.load(f)

    # Create model
    print("Loading model...")
    model_type = config['model']['type']
    if model_type == 'vit':
        model_config = config['model']['vit']
    elif model_type == 'lstm':
        model_config = config['model']['lstm']
    else:
        model_config = {}

    model = get_model(model_type, num_classes, model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    print(f"Evaluating on {args.split} set...")
    y_pred, y_true, y_probs = evaluate_model(model, dataloader, device, class_names)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"Results on {args.split.upper()} set")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Get unique classes present in predictions
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    present_class_names = [class_names[i] for i in unique_classes]

    # Warn if some classes are missing
    if len(unique_classes) < len(class_names):
        missing_classes = set(range(len(class_names))) - set(unique_classes)
        missing_names = [class_names[i] for i in missing_classes]
        print(f"\n⚠️  WARNING: Only {len(unique_classes)}/{len(class_names)} classes present in {args.split} set")
        print(f"   Present classes: {present_class_names}")
        print(f"   Missing classes: {missing_names}")
        print(f"   This indicates your dataset is too small or imbalanced!\n")

    # Classification report - use only present classes
    report = classification_report(
        y_true, y_pred,
        labels=unique_classes.tolist(),
        target_names=present_class_names,
        digits=4
    )
    print("\nClassification Report:")
    print(report)

    # Save report
    with open(output_dir / f'{args.split}_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    # Plot confusion matrix - use only present classes
    plot_confusion_matrix(
        y_true, y_pred, present_class_names,
        output_dir / f'{args.split}_confusion_matrix.png'
    )

    # Save predictions
    results = {
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_probs': y_probs.tolist(),
        'class_names': class_names,
        'accuracy': float(accuracy)
    }

    with open(output_dir / f'{args.split}_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()