#!/usr/bin/env python
"""
Script 02: Train U-Net segmentation model with comprehensive logging and early stopping
"""
import os
import sys
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import path_config, seg_config
from src.dataset import create_segmentation_dataloaders
from src.models import SegmentationModel
from src.training import SegmentationTrainer


def visualize_training_history(history_path, save_dir=None):
    """
    Visualize training history from saved CSV/JSON without retraining
    """
    import pandas as pd
    
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
    else:
        # Try JSON format
        json_path = history_path.replace('.csv', '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            print(f"No history file found at {history_path}")
            return
    
    if save_dir is None:
        save_dir = os.path.dirname(history_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss curves
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. IoU curve
    axes[0, 1].plot(df['epoch'], df['val_iou'], label='Validation IoU', color='green', linewidth=2)
    axes[0, 1].fill_between(df['epoch'], 0, df['val_iou'], alpha=0.3, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Dice curve
    if 'val_dice' in df.columns:
        axes[1, 0].plot(df['epoch'], df['val_dice'], label='Validation Dice', color='purple', linewidth=2)
        axes[1, 0].fill_between(df['epoch'], 0, df['val_dice'], alpha=0.3, color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice')
        axes[1, 0].set_title('Validation Dice Coefficient')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Dice data not available', ha='center', va='center')
        axes[1, 0].set_title('Validation Dice (Not Available)')
    
    # 4. Pixel Accuracy
    if 'val_pixel_acc' in df.columns:
        axes[1, 1].plot(df['epoch'], df['val_pixel_acc'], label='Pixel Accuracy', color='orange', linewidth=2)
        axes[1, 1].fill_between(df['epoch'], 0, df['val_pixel_acc'], alpha=0.3, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Pixel Accuracy')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Pixel Accuracy data not available', ha='center', va='center')
        axes[1, 1].set_title('Pixel Accuracy (Not Available)')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")
    
    # Also create a combined metrics plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['val_iou'], label='IoU', marker='o', linewidth=2)
    if 'val_dice' in df.columns:
        ax.plot(df['epoch'], df['val_dice'], label='Dice', marker='s', linewidth=2)
    if 'val_pixel_acc' in df.columns:
        ax.plot(df['epoch'], df['val_pixel_acc'], label='Pixel Accuracy', marker='^', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Metrics Over Time')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    metrics_path = os.path.join(save_dir, 'validation_metrics.png')
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Validation metrics saved to {metrics_path}")
    
    return df


def main():
    """Train segmentation model with comprehensive logging and early stopping"""
    
    print("=" * 60)
    print("Defect Detection - Segmentation Training")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs(path_config.segmentation_models, exist_ok=True)
    os.makedirs('./logs/tensorboard', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Early stopping configuration
    EARLY_STOPPING_PATIENCE = 15  # Number of epochs to wait for improvement
    MIN_DELTA = 0.001  # Minimum change to qualify as improvement
    EARLY_STOPPING_MODE = 'min'  # 'min' for loss, 'max' for metrics
    
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Early stopping min delta: {MIN_DELTA}")
    print(f"Early stopping mode: {EARLY_STOPPING_MODE}")
    
    # Save training config
    config_info = {
        'encoder_name': seg_config.encoder_name,
        'input_size': seg_config.input_size,
        'batch_size': seg_config.batch_size,
        'num_epochs': seg_config.num_epochs,
        'learning_rate': seg_config.learning_rate,
        'weight_decay': seg_config.weight_decay,
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'early_stopping_min_delta': MIN_DELTA,
        'early_stopping_mode': EARLY_STOPPING_MODE
    }
    
    config_path = os.path.join(path_config.segmentation_models, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)
    print(f"Training config saved to {config_path}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_segmentation_dataloaders(
        data_root=path_config.data_preprocessed,
        batch_size=seg_config.batch_size,
        input_size=seg_config.input_size,
        train_ratio=0.8
    )
    
    # Create model
    print("\nCreating model...")
    model = SegmentationModel(
        encoder_name=seg_config.encoder_name,
        num_classes=seg_config.num_classes,
        in_channels=seg_config.in_channels
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=seg_config.num_epochs,
        lr=seg_config.learning_rate,
        model_name='unet_efficientnet_b0',
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_mode=EARLY_STOPPING_MODE,
        min_delta=MIN_DELTA
    )
    
    train_losses, val_losses, val_mious = trainer.train()
    
    # Visualize training history from saved data
    history_path = os.path.join(path_config.segmentation_models, 'training_history.csv')
    if os.path.exists(history_path):
        print("\n" + "=" * 60)
        print("Generating training visualizations...")
        print("=" * 60)
        visualize_training_history(history_path, path_config.segmentation_models)
    
    # Print early stopping info
    early_stop_summary_path = os.path.join(path_config.segmentation_models, 'early_stopping_summary.json')
    if os.path.exists(early_stop_summary_path):
        with open(early_stop_summary_path, 'r') as f:
            early_summary = json.load(f)
        print("\n" + "=" * 60)
        print("Early Stopping Information")
        print("=" * 60)
        print(f"Early stop triggered: {early_summary.get('early_stop_triggered', False)}")
        if early_summary.get('early_stop_triggered', False):
            print(f"Stopped at epoch: {early_summary.get('stopped_epoch', 'N/A')}")
            print(f"Best score: {early_summary.get('best_score', 'N/A'):.6f}")
            print(f"Best epoch: {early_summary.get('best_epoch', 'N/A')}")
            print(f"Patience used: {early_summary.get('patience_used', 0)}/{early_summary.get('patience_max', 0)}")
    
    print("\n✅ Training completed!")
    print(f"Models saved to: {path_config.segmentation_models}")
    print(f"Logs saved to: ./logs/")


if __name__ == "__main__":
    main()