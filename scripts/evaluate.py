#!/usr/bin/env python
"""
Script 07: Evaluate models and generate results for report
"""
import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import path_config, seg_config
from src.models import SegmentationModel
from src.metrics import DetectionMetrics, SegmentationMetrics, EvaluationLogger


class ModelEvaluator:
    """Evaluate segmentation and detection models"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.seg_model = None
        self.eval_logger = EvaluationLogger()
        
    def load_model(self, model_path):
        """Load trained segmentation model"""
        self.seg_model = SegmentationModel(
            encoder_name=seg_config.encoder_name,
            num_classes=seg_config.num_classes,
            in_channels=seg_config.in_channels
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        self.seg_model.load_state_dict(checkpoint['model_state_dict'])
        self.seg_model = self.seg_model.to(self.device)
        self.seg_model.eval()
        print(f"Loaded model from {model_path}")
    
    def evaluate_on_dataset(self, image_paths, mask_paths, save_error_analysis=True):
        """Evaluate model on a dataset"""
        
        seg_metrics = []
        det_predictions = []
        det_groundtruths = []
        error_cases = []
        
        for img_path, mask_path in tqdm(zip(image_paths, mask_paths), 
                                        total=len(image_paths), 
                                        desc="Evaluating"):
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Get prediction mask
            pred_mask = self._predict_mask(image)
            
            # Load ground truth mask
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                continue
            
            # Resize gt_mask to match pred_mask if needed
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # Calculate segmentation metrics
            metrics = SegmentationMetrics.calculate_all_metrics(pred_mask, gt_mask)
            seg_metrics.append(metrics)
            
            # Extract detection results
            pred_boxes = self._extract_boxes_from_mask(pred_mask)
            gt_boxes = self._extract_boxes_from_mask(gt_mask)
            
            det_predictions.append({'boxes': pred_boxes, 'scores': [1.0] * len(pred_boxes)})
            det_groundtruths.append({'boxes': gt_boxes})
            
            # Collect error cases for analysis
            if metrics['iou'] < 0.5:  # Low IoU cases
                error_cases.append({
                    'image_path': img_path,
                    'gt_mask': gt_mask,
                    'pred_mask': pred_mask,
                    'iou': metrics['iou'],
                    'dice': metrics['dice']
                })
        
        # Calculate average metrics
        avg_metrics = {
            'iou': np.mean([m['iou'] for m in seg_metrics]),
            'dice': np.mean([m['dice'] for m in seg_metrics]),
            'pixel_accuracy': np.mean([m['pixel_accuracy'] for m in seg_metrics])
        }
        
        # Calculate detection metrics
        precision, recall = DetectionMetrics.calculate_precision_recall(
            det_predictions, det_groundtruths
        )
        map_score = DetectionMetrics.calculate_map(det_predictions, det_groundtruths)
        
        # Log results
        self.eval_logger.add_segmentation_result(
            name='U-Net + EfficientNet-B0',
            iou=avg_metrics['iou'],
            dice=avg_metrics['dice'],
            pixel_accuracy=avg_metrics['pixel_accuracy']
        )
        
        self.eval_logger.add_detection_result(
            name='U-Net + EfficientNet-B0',
            map_score=map_score,
            precision=precision,
            recall=recall
        )
        
        if save_error_analysis:
            self._save_error_analysis(error_cases)
        
        return avg_metrics, map_score, precision, recall, error_cases
    
    def _predict_mask(self, image):
        """Get prediction mask from model"""
        # Preprocess image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray = cv2.resize(gray, seg_config.input_size)
        input_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.seg_model(input_tensor)
            pred = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        
        return (pred > 0.5).astype(np.uint8)
    
    def _extract_boxes_from_mask(self, mask):
        """Extract bounding boxes from mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:  # Min area threshold
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([x, y, x + w, y + h])
        return boxes
    
    def _save_error_analysis(self, error_cases, max_cases=10):
        """Save error cases for analysis"""
        error_dir = './logs/evaluation/error_analysis'
        os.makedirs(error_dir, exist_ok=True)
        
        for i, case in enumerate(error_cases[:max_cases]):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Show image
            img = cv2.imread(case['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Show ground truth
            axes[1].imshow(case['gt_mask'], cmap='gray')
            axes[1].set_title(f'Ground Truth (IoU: {case["iou"]:.3f})')
            axes[1].axis('off')
            
            # Show prediction
            axes[2].imshow(case['pred_mask'], cmap='gray')
            axes[2].set_title(f'Prediction (Dice: {case["dice"]:.3f})')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(error_dir, f'error_case_{i+1}.png'), dpi=150)
            plt.close()
        
        print(f"Saved {len(error_cases[:max_cases])} error cases to {error_dir}")


def compare_baseline_vs_improved(baseline_results, improved_results):
    """Generate comparison table for report"""
    
    comparison = {
        'Metric': ['mAP', 'Precision', 'Recall', 'IoU', 'Dice', 'Pixel Accuracy'],
        'Baseline': [
            baseline_results.get('map', 0),
            baseline_results.get('precision', 0),
            baseline_results.get('recall', 0),
            baseline_results.get('iou', 0),
            baseline_results.get('dice', 0),
            baseline_results.get('pixel_accuracy', 0)
        ],
        'Improved': [
            improved_results.get('map', 0),
            improved_results.get('precision', 0),
            improved_results.get('recall', 0),
            improved_results.get('iou', 0),
            improved_results.get('dice', 0),
            improved_results.get('pixel_accuracy', 0)
        ]
    }
    
    # Calculate improvements
    improvements = []
    for i in range(len(comparison['Baseline'])):
        if comparison['Baseline'][i] != 0:
            imp = (comparison['Improved'][i] - comparison['Baseline'][i]) / comparison['Baseline'][i] * 100
        else:
            imp = comparison['Improved'][i] * 100
        improvements.append(f"{imp:+.1f}%")
    
    comparison['Improvement'] = improvements
    
    import pandas as pd
    df = pd.DataFrame(comparison)
    return df


def generate_report_plots(training_logger):
    """Generate all plots needed for report"""
    
    # 1. Training curves
    training_logger.plot_training_curves()
    
    # 2. Confusion matrix-like analysis
    history = pd.read_csv(os.path.join(training_logger.log_dir, 'training_history.csv'))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss convergence
    axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train')
    axes[0, 0].plot(history['epoch'], history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU improvement
    axes[0, 1].plot(history['epoch'], history['val_iou'], color='green')
    axes[0, 1].fill_between(history['epoch'], 0, history['val_iou'], alpha=0.3, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('IoU Improvement Over Time')
    axes[0, 1].grid(True)
    
    # Dice coefficient
    axes[1, 0].plot(history['epoch'], history['val_dice'], color='purple')
    axes[1, 0].fill_between(history['epoch'], 0, history['val_dice'], alpha=0.3, color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].set_title('Dice Coefficient Over Time')
    axes[1, 0].grid(True)
    
    # Learning rate schedule
    axes[1, 1].plot(history['epoch'], history['learning_rate'], color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('./logs/evaluation/report_plots.png', dpi=150)
    plt.close()


def main():
    """Main evaluation script"""
    
    print("=" * 60)
    print("Model Evaluation for Report")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load trained model
    model_path = os.path.join(path_config.segmentation_models, 'best_model.pth')
    if os.path.exists(model_path):
        evaluator.load_model(model_path)
    else:
        print(f"Model not found at {model_path}")
        return
    
    # TODO: Load test dataset paths
    # test_images = [...]
    # test_masks = [...]
    
    # Evaluate
    # metrics, map_score, precision, recall, error_cases = evaluator.evaluate_on_dataset(
    #     test_images, test_masks
    # )
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()