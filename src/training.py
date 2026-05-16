"""
Training utilities for segmentation and VQA
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from datetime import datetime
import json

from .config import path_config
from .metrics import SegmentationMetrics
from .training_logger import TrainingLogger


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience (int): Number of epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as an improvement.
        mode (str): 'min' for loss, 'max' for metrics like IoU.
        verbose (bool): Print messages when stopping.
        save_best (bool): Save best model when improvement occurs.
    """
    def __init__(self, patience=10, min_delta=0.000001, mode='min', verbose=True, save_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_best = save_best
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # Determine if we're maximizing or minimizing
        if self.mode == 'min':
            self.is_better = lambda current, best: current < best - self.min_delta
        elif self.mode == 'max':
            self.is_better = lambda current, best: current > best + self.min_delta
        else:
            raise ValueError(f"Mode '{mode}' not recognized. Use 'min' or 'max'.")
    
    def __call__(self, score, epoch, model=None, optimizer=None, scheduler=None, save_path=None):
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (loss or metric)
            epoch: Current epoch number
            model: Model to save (optional)
            optimizer: Optimizer to save (optional)
            scheduler: Scheduler to save (optional)
            save_path: Path to save checkpoint (optional)
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            # First epoch: set best score and save if requested
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"Initial best score: {self.best_score:.6f}")
            if self.save_best and model is not None and save_path is not None:
                self._save_checkpoint(model, optimizer, scheduler, epoch, score, save_path)
            return False
        
        if self.is_better(score, self.best_score):
            # Improvement detected: reset counter and update best
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  ✅ Improvement! New best score: {self.best_score:.6f}")
            if self.save_best and model is not None and save_path is not None:
                self._save_checkpoint(model, optimizer, scheduler, epoch, score, save_path)
        else:
            # No improvement: increment counter
            self.counter += 1
            if self.verbose:
                print(f"  No improvement. Patience: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n⚠️ Early stopping triggered! No improvement for {self.patience} epochs.")
                    print(f"   Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
        
        return self.early_stop
    
    def _save_checkpoint(self, model, optimizer, scheduler, epoch, score, save_path):
        """Save checkpoint when improvement occurs"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_score': score,
            'best_epoch': epoch
        }
        torch.save(checkpoint, save_path)
        if self.verbose:
            print(f"  💾 Best model saved to: {save_path}")
    
    def get_summary(self):
        """Get early stopping summary"""
        return {
            'early_stop': self.early_stop,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'patience_used': self.counter,
            'patience_max': self.patience
        }


class SegmentationTrainer:
    """Trainer for U-Net segmentation with comprehensive logging and early stopping"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 num_epochs=3, lr=1e-4, model_name='segmentation',
                 early_stopping_patience=10, early_stopping_mode='min',
                 min_delta=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.model_name = model_name
        self.early_stop_triggered = False
        self.stopped_epoch = None
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=min_delta,
            mode=early_stopping_mode,
            verbose=True,
            save_best=True
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        self.val_dices = []
        self.val_pixel_accs = []
        
        # Initialize logger
        self.logger = TrainingLogger(log_dir='./logs/training', model_name=model_name)
        
        # For TensorBoard
        self.writer = None
        self._setup_tensorboard()
    
    def _setup_tensorboard(self):
        """Setup TensorBoard writer"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join('./logs/tensorboard', self.model_name, 
                                   datetime.now().strftime('%Y%m%d_%H%M%S'))
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard enabled. Logs will be saved to: {log_dir}")
            print(f"Run: tensorboard --logdir={log_dir}")
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
    
    def calculate_all_metrics(self, pred, target):
        """Calculate all segmentation metrics"""
        # Convert to numpy for metric calculation
        pred_softmax = torch.softmax(pred, dim=1)
        pred_mask = torch.argmax(pred_softmax, dim=1).cpu().numpy()
        target_mask = target.cpu().numpy()
        
        iou = SegmentationMetrics.calculate_iou(pred_mask, target_mask)
        dice = SegmentationMetrics.calculate_dice(pred_mask, target_mask)
        pixel_acc = SegmentationMetrics.calculate_pixel_accuracy(pred_mask, target_mask)
        
        return iou, dice, pixel_acc
    
    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        
        loop = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
            # Log to TensorBoard (every 10 batches)
            if self.writer and batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        val_pixel_acc = 0
        
        with torch.no_grad():
            loop = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')
            for images, masks in loop:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate all metrics
                iou, dice, pixel_acc = self.calculate_all_metrics(outputs, masks)
                
                val_loss += loss.item()
                val_iou += iou
                val_dice += dice
                val_pixel_acc += pixel_acc
                
                loop.set_postfix(loss=loss.item(), iou=f'{iou:.4f}', dice=f'{dice:.4f}')
        
        n_batches = len(self.val_loader)
        avg_loss = val_loss / n_batches
        avg_iou = val_iou / n_batches
        avg_dice = val_dice / n_batches
        avg_pixel_acc = val_pixel_acc / n_batches
        
        # Save best model by loss
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_checkpoint(epoch, avg_loss, avg_iou, is_best=True)
        
        # Save best model by IoU
        if avg_iou > self.best_val_iou:
            self.best_val_iou = avg_iou
            self._save_checkpoint(epoch, avg_loss, avg_iou, is_best_iou=True)
        
        # Log to logger
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.log_epoch(epoch + 1, 
                             self.train_losses[-1] if self.train_losses else 0,
                             avg_loss, avg_iou, avg_dice, avg_pixel_acc, current_lr, 0)
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
            self.writer.add_scalar('Validation/IoU', avg_iou, epoch)
            self.writer.add_scalar('Validation/Dice', avg_dice, epoch)
            self.writer.add_scalar('Validation/Pixel_Accuracy', avg_pixel_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Update scheduler
        self.scheduler.step(avg_loss)
        
        return avg_loss, avg_iou, avg_dice, avg_pixel_acc
    
    def _save_checkpoint(self, epoch, val_loss, val_iou, is_best=False, is_best_iou=False):
        """Save model checkpoint"""
        os.makedirs(path_config.segmentation_models, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_iou': val_iou,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious
        }
        
        if is_best:
            save_path = os.path.join(path_config.segmentation_models, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved best model (by loss) with val_loss: {val_loss:.4f}, mIoU: {val_iou:.4f}")
        
        if is_best_iou:
            save_path = os.path.join(path_config.segmentation_models, 'best_iou_model.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved best model (by IoU) with val_loss: {val_loss:.4f}, mIoU: {val_iou:.4f}")
        
        # Save latest checkpoint
        latest_path = os.path.join(path_config.segmentation_models, 'latest_model.pth')
        torch.save(checkpoint, latest_path)
    
    def _save_early_stop_checkpoint(self, epoch, val_loss, val_iou):
        """Save final checkpoint when early stopping occurs"""
        checkpoint_path = os.path.join(path_config.segmentation_models, 'early_stop_model.pth')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_iou': val_iou,
            'early_stop_epoch': epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"\nEarly stop model saved to: {checkpoint_path}")
    
    def train(self):
        """Main training loop with early stopping"""
        print("=" * 60)
        print(f"Starting {self.model_name} training...")
        print(f"Device: {self.device}")
        print(f"Num epochs: {self.num_epochs}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Early stopping patience: {self.early_stopping.patience}")
        print("=" * 60)
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_one_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_iou, val_dice, val_pixel_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_mious.append(val_iou)
            self.val_dices.append(val_dice)
            self.val_pixel_accs.append(val_pixel_acc)
            
            # Print summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val IoU: {val_iou:.4f}")
            print(f"  Val Dice: {val_dice:.4f}")
            print(f"  Val Pixel Acc: {val_pixel_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"{'='*50}\n")
            
            # Early stopping check (using validation loss)
            save_path = os.path.join(path_config.segmentation_models, 'best_model.pth')
            should_stop = self.early_stopping(val_loss, epoch + 1, self.model, 
                                              self.optimizer, self.scheduler, save_path)
            
            if should_stop:
                self.early_stop_triggered = True
                self.stopped_epoch = epoch + 1
                self._save_early_stop_checkpoint(epoch, val_loss, val_iou)
                break
        
        # Save training history
        self._save_training_history()
        
        # Save early stopping summary
        self._save_early_stopping_summary()
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        # Print final summary
        self._print_final_summary()
        
        return self.train_losses, self.val_losses, self.val_mious
    
    def _save_training_history(self):
        """Save training history to CSV and JSON"""
        import pandas as pd
        
        history_df = pd.DataFrame({
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_iou': self.val_mious,
            'val_dice': self.val_dices,
            'val_pixel_acc': self.val_pixel_accs
        })
        
        csv_path = os.path.join(path_config.segmentation_models, 'training_history.csv')
        history_df.to_csv(csv_path, index=False)
        print(f"Training history saved to {csv_path}")
        
        # Save as JSON for easy loading
        json_path = os.path.join(path_config.segmentation_models, 'training_history.json')
        with open(json_path, 'w') as f:
            json.dump(history_df.to_dict(orient='list'), f, indent=2)
    
    def _save_early_stopping_summary(self):
        """Save early stopping summary"""
        summary = self.early_stopping.get_summary()
        summary['early_stop_triggered'] = self.early_stop_triggered
        summary['stopped_epoch'] = self.stopped_epoch
        summary['total_epochs_completed'] = len(self.train_losses)
        
        summary_path = os.path.join(path_config.segmentation_models, 'early_stopping_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Early stopping summary saved to {summary_path}")
    
    def _print_final_summary(self):
        """Print final training summary"""
        print("\n" + "=" * 60)
        if self.early_stop_triggered:
            print("TRAINING STOPPED EARLY")
            print(f"Reason: No improvement for {self.early_stopping.patience} epochs")
            print(f"Stopped at epoch: {self.stopped_epoch}")
        else:
            print("TRAINING COMPLETED")
            print(f"Completed all {len(self.train_losses)} epochs")
        print("=" * 60)
        print(f"Best Validation Loss: {self.best_val_loss:.6f}")
        print(f"Best Validation IoU: {self.best_val_iou:.4f}")
        print(f"Best Validation Dice: {max(self.val_dices):.4f}")
        print(f"Best Validation Pixel Accuracy: {max(self.val_pixel_accs):.4f}")
        print(f"Model saved to: {path_config.segmentation_models}")
        
        if self.early_stop_triggered:
            print(f"\n⏹️ Early stopping summary:")
            print(f"   Best score: {self.early_stopping.best_score:.6f}")
            print(f"   Best epoch: {self.early_stopping.best_epoch}")
            print(f"   Patience used: {self.early_stopping.counter}/{self.early_stopping.patience}")
        print("=" * 60)