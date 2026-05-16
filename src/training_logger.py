"""
Training logger for tracking fine-tuning process
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict


class TrainingLogger:
    """Log training metrics for report generation"""
    
    def __init__(self, log_dir='./logs/training', model_name='segmentation'):
        self.log_dir = os.path.join(log_dir, model_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.model_name = model_name
        self.start_time = datetime.now()
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_dice': [],
            'val_pixel_acc': [],
            'learning_rate': [],
            'time_per_epoch': []
        }
        
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_val_iou': 0.0,
            'best_val_dice': 0.0,
            'best_epoch': 0
        }
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  val_iou: float, val_dice: float, val_pixel_acc: float,
                  lr: float, epoch_time: float):
        """Log metrics for one epoch"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_iou'].append(val_iou)
        self.history['val_dice'].append(val_dice)
        self.history['val_pixel_acc'].append(val_pixel_acc)
        self.history['learning_rate'].append(lr)
        self.history['time_per_epoch'].append(epoch_time)
        
        # Update best metrics
        if val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = val_loss
            self.best_metrics['best_epoch'] = epoch
        
        if val_iou > self.best_metrics['best_val_iou']:
            self.best_metrics['best_val_iou'] = val_iou
        
        if val_dice > self.best_metrics['best_val_dice']:
            self.best_metrics['best_val_dice'] = val_dice
        
        # Save after each epoch
        self.save_history()
    
    def save_history(self):
        """Save training history to CSV"""
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.log_dir, 'training_history.csv'), index=False)
        
        # Save best metrics
        with open(os.path.join(self.log_dir, 'best_metrics.json'), 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], 
                        label='Train Loss', color='blue', linewidth=2)
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], 
                        label='Val Loss', color='red', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # IoU curve
        axes[0, 1].plot(self.history['epoch'], self.history['val_iou'], 
                        label='Validation IoU', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('Validation IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Dice curve
        axes[1, 0].plot(self.history['epoch'], self.history['val_dice'], 
                        label='Validation Dice', color='purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice')
        axes[1, 0].set_title('Validation Dice Coefficient')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate curve
        axes[1, 1].plot(self.history['epoch'], self.history['learning_rate'], 
                        label='Learning Rate', color='orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training curves to {save_path}")
    
    def get_summary(self) -> Dict:
        """Get training summary"""
        return {
            'model_name': self.model_name,
            'training_duration': str(datetime.now() - self.start_time),
            'num_epochs': len(self.history['epoch']),
            'best_epoch': self.best_metrics['best_epoch'],
            'best_val_loss': self.best_metrics['best_val_loss'],
            'best_val_iou': self.best_metrics['best_val_iou'],
            'best_val_dice': self.best_metrics['best_val_dice'],
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'final_val_iou': self.history['val_iou'][-1] if self.history['val_iou'] else None
        }


class VQATrainingLogger:
    """Logger specifically for VQA fine-tuning"""
    
    def __init__(self, log_dir='./logs/training/vqa'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def log_epoch(self, epoch, train_loss, val_loss, val_acc, lr):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_acc)
        self.history['learning_rate'].append(lr)
        self.save_history()
    
    def save_history(self):
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.log_dir, 'vqa_training_history.csv'), index=False)
    
    def plot_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('VQA Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history['epoch'], self.history['val_accuracy'], 
                     label='Val Accuracy', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('VQA Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'vqa_training_curves.png'), dpi=150)
        plt.close()