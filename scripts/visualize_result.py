#!/usr/bin/env python
"""
Script 10: Visualize training results from saved logs (without retraining)
"""
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import path_config


def find_latest_training_history():
    """Find the latest training history file"""
    possible_paths = [
        os.path.join(path_config.segmentation_models, 'training_history.csv'),
        os.path.join(path_config.segmentation_models, 'training_history.json'),
        './logs/training/segmentation/training_history.csv',
        './logs/training/segmentation/training_history.json'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def load_training_data(file_path):
    """Load training data from CSV or JSON"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        return None


def plot_loss_curves(df, save_path):
    """Plot loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to: {save_path}")


def plot_metric_curves(df, metric_name, save_path):
    """Plot a single metric curve"""
    plt.figure(figsize=(10, 6))
    if metric_name in df.columns:
        plt.plot(df['epoch'], df[metric_name], linewidth=2)
        plt.fill_between(df['epoch'], 0, df[metric_name], alpha=0.3)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'Validation {metric_name} Over Time')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"{metric_name} chart saved to: {save_path}")
    else:
        print(f"Metric {metric_name} not found in data")


def plot_all_metrics(df, save_path):
    """Plot all metrics in one figure"""
    metrics = ['val_iou', 'val_dice', 'val_pixel_acc']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print("No metrics found for plotting")
        return
    
    plt.figure(figsize=(12, 6))
    for metric in available_metrics:
        label = metric.replace('val_', '').replace('_', ' ').title()
        plt.plot(df['epoch'], df[metric], label=label, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined metrics chart saved to: {save_path}")


def print_summary(df):
    """Print summary statistics"""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    # Best values
    best_loss_idx = df['val_loss'].idxmin()
    best_iou_idx = df['val_iou'].idxmax() if 'val_iou' in df.columns else None
    
    print(f"Total epochs: {len(df)}")
    print(f"Best Validation Loss: {df['val_loss'].min():.6f} (epoch {best_loss_idx + 1})")
    
    if best_iou_idx is not None:
        print(f"Best Validation IoU: {df['val_iou'].max():.4f} (epoch {best_iou_idx + 1})")
    
    if 'val_dice' in df.columns:
        print(f"Best Validation Dice: {df['val_dice'].max():.4f}")
    
    if 'val_pixel_acc' in df.columns:
        print(f"Best Pixel Accuracy: {df['val_pixel_acc'].max():.4f}")
    
    # Final values
    print(f"\nFinal Validation Loss: {df['val_loss'].iloc[-1]:.6f}")
    if 'val_iou' in df.columns:
        print(f"Final Validation IoU: {df['val_iou'].iloc[-1]:.4f}")
    
    print("=" * 60)


def generate_visualization_report(df, output_dir):
    """Generate all visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss curves
    plot_loss_curves(df, os.path.join(output_dir, 'loss_curves.png'))
    
    # Plot individual metrics
    plot_metric_curves(df, 'val_iou', os.path.join(output_dir, 'iou_curve.png'))
    plot_metric_curves(df, 'val_dice', os.path.join(output_dir, 'dice_curve.png'))
    plot_metric_curves(df, 'val_pixel_acc', os.path.join(output_dir, 'pixel_accuracy_curve.png'))
    
    # Plot combined metrics
    plot_all_metrics(df, os.path.join(output_dir, 'all_metrics.png'))
    
    # Print summary
    print_summary(df)
    
    # Save summary to file
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total epochs: {len(df)}\n")
        f.write(f"Best Validation Loss: {df['val_loss'].min():.6f}\n")
        if 'val_iou' in df.columns:
            f.write(f"Best Validation IoU: {df['val_iou'].max():.4f}\n")
        if 'val_dice' in df.columns:
            f.write(f"Best Validation Dice: {df['val_dice'].max():.4f}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nSummary saved to: {summary_path}")


def main():
    """Main visualization function"""
    print("=" * 60)
    print("Training Results Visualizer")
    print("=" * 60)
    
    # Find training history
    history_path = find_latest_training_history()
    
    if history_path is None:
        print("No training history found. Please train the model first.")
        print("Run: python scripts/02_train_segmentation.py")
        return
    
    print(f"Loading training data from: {history_path}")
    df = load_training_data(history_path)
    
    if df is None:
        print("Failed to load training data")
        return
    
    print(f"Loaded {len(df)} epochs")
    
    # Generate visualizations
    output_dir = './logs/visualizations'
    generate_visualization_report(df, output_dir)
    
    print("\n✅ Visualization completed!")
    print(f"All charts saved to: {output_dir}")
    
    # Instructions for TensorBoard
    tensorboard_dirs = glob.glob('./logs/tensorboard/*/*')
    if tensorboard_dirs:
        print("\n" + "=" * 60)
        print("TensorBoard Instructions")
        print("=" * 60)
        print(f"To view real-time training curves with TensorBoard, run:")
        print(f"    tensorboard --logdir=./logs/tensorboard")
        print(f"\nThen open http://localhost:6006 in your browser")
        print("=" * 60)


if __name__ == "__main__":
    main()