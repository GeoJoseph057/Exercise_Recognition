"""
Training Script for Exercise Recognition
Supports LSTM, ViT, and Hybrid models with full training pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

# Import custom modules (adjust paths as needed)
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.pose_encoder import get_model
from utils.dataset import create_dataloaders


class Trainer:
    """Training manager with all the bells and whistles"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create directories
        self.exp_dir = Path(config['paths']['checkpoints']) / config['experiment_name']
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.writer = SummaryWriter(
            Path(config['paths']['tensorboard']) / config['experiment_name']
        )

        # Save config
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

        # Initialize metrics tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def setup_data(self):
        """Setup dataloaders"""
        print("\n📊 Setting up dataloaders...")

        self.train_loader, self.val_loader, self.test_loader, self.num_classes = \
            create_dataloaders(
                data_dir=self.config['data']['processed_path'],
                batch_size=self.config['training']['batch_size'],
                sequence_length=self.config['pose']['temporal_window'],
                num_workers=0
            )

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
        print(f"Number of classes: {self.num_classes}")

    def setup_model(self):
        """Initialize model, optimizer, scheduler"""
        print("\n🤖 Setting up model...")

        # Get model config based on type
        model_type = self.config['model']['type']
        if model_type == 'vit':
            model_config = self.config['model']['vit']
        elif model_type == 'lstm':
            model_config = self.config['model']['lstm']
        else:
            model_config = {}

        # Create model
        self.model = get_model(
            model_type=model_type,
            num_classes=self.num_classes,
            config=model_config
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        wd = self.config['training']['weight_decay']

        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                      weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Learning rate scheduler
        scheduler_name = self.config['training']['scheduler'].lower()
        total_epochs = self.config['training']['epochs']
        warmup_epochs = self.config['training']['warmup_epochs']

        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs - warmup_epochs
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=total_epochs // 3, gamma=0.1
            )
        else:
            self.scheduler = None

        print(f"Optimizer: {optimizer_name}")
        print(f"Scheduler: {scheduler_name}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (poses, labels) in enumerate(pbar):
            poses, labels = poses.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(poses)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

            # Log to tensorboard
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss_Step', loss.item(), step)

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for poses, labels in tqdm(self.val_loader, desc="Validating"):
                poses, labels = poses.to(self.device), labels.to(self.device)

                outputs = self.model(poses)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc, all_preds, all_labels

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }

        # Save last checkpoint
        torch.save(checkpoint, self.exp_dir / 'last.pth')

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.exp_dir / 'best.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")

    def train(self):
        """Full training loop"""
        print("\n🚀 Starting training...")
        print(f"Experiment: {self.config['experiment_name']}")
        print(f"Total epochs: {self.config['training']['epochs']}")

        patience = self.config['training']['patience']
        epochs_no_improve = 0

        for epoch in range(self.config['training']['epochs']):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"{'='*60}")

            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc, preds, labels = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Logging
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            self.writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate',
                                  self.optimizer.param_groups[0]['lr'], epoch)

            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_acc, is_best)

            # Learning rate scheduling
            if self.scheduler and epoch >= self.config['training']['warmup_epochs']:
                self.scheduler.step()

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                print(f"Best Val Acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")
                break

        print(f"\n✅ Training completed!")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
        with open(self.exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Exercise Recognition Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, choices=['lstm', 'vit', 'hybrid'],
                       help='Model type (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    if args.model:
        config['model']['type'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # Add timestamp to experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['experiment_name'] = f"{config['experiment_name']}_{timestamp}"

    # Initialize trainer
    trainer = Trainer(config)

    # Setup
    trainer.setup_data()
    trainer.setup_model()

    # Train
    trainer.train()


if __name__ == '__main__':
    main()