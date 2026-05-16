"""
Metrics for detection and segmentation evaluation
"""
import os

import numpy as np
import torch
from typing import List, Dict, Tuple
import pandas as pd


class DetectionMetrics:
    """Calculate detection metrics: mAP, Precision, Recall"""
    
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    @staticmethod
    def calculate_ap(precisions, recalls):
        """Calculate Average Precision"""
        if len(precisions) == 0 or len(recalls) == 0:
            return 0.0
        
        # Add sentinel values
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        # Compute AP
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
        indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
        
        return ap
    
    @staticmethod
    def calculate_map(predictions: List[Dict], groundtruths: List[Dict], iou_threshold=0.5):
        """
        Calculate mAP for defect detection
        
        Args:
            predictions: List of dicts with 'boxes' and 'scores'
            groundtruths: List of dicts with 'boxes'
            iou_threshold: IoU threshold for positive detection
        """
        all_aps = []
        
        for cls_name in ['defect']:
            all_pred_boxes = []
            all_pred_scores = []
            all_gt_boxes = []
            
            for pred, gt in zip(predictions, groundtruths):
                all_pred_boxes.append(pred.get('boxes', []))
                all_pred_scores.append(pred.get('scores', []))
                all_gt_boxes.append(gt.get('boxes', []))
            
            ap = DetectionMetrics._calculate_class_ap(
                all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold
            )
            all_aps.append(ap)
        
        return np.mean(all_aps)
    
    @staticmethod
    def _calculate_class_ap(pred_boxes_list, pred_scores_list, gt_boxes_list, iou_threshold):
        """Calculate AP for a single class"""
        all_pred = []
        all_gt = []
        
        for pred_boxes, pred_scores, gt_boxes in zip(pred_boxes_list, pred_scores_list, gt_boxes_list):
            for box, score in zip(pred_boxes, pred_scores):
                all_pred.append({
                    'box': box,
                    'score': score,
                    'matched': False
                })
            for gt in gt_boxes:
                all_gt.append({'box': gt, 'matched': False})
        
        # Sort predictions by score
        all_pred.sort(key=lambda x: x['score'], reverse=True)
        
        tp = np.zeros(len(all_pred))
        fp = np.zeros(len(all_pred))
        
        for i, pred in enumerate(all_pred):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(all_gt):
                if not gt['matched']:
                    iou = DetectionMetrics.calculate_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp[i] = 1
                all_gt[best_gt_idx]['matched'] = True
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / (len(all_gt) + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        return DetectionMetrics.calculate_ap(precisions, recalls)
    
    @staticmethod
    def calculate_precision_recall(predictions, groundtruths, iou_threshold=0.5):
        """Calculate Precision and Recall"""
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        
        for pred, gt in zip(predictions, groundtruths):
            all_pred_boxes.extend(pred.get('boxes', []))
            all_pred_scores.extend(pred.get('scores', []))
            all_gt_boxes.extend(gt.get('boxes', []))
        
        if len(all_pred_boxes) == 0:
            return 0.0, 0.0
        
        # Match predictions to ground truths
        matched_pred = 0
        matched_gt = set()
        
        for i, pred_box in enumerate(all_pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(all_gt_boxes):
                if j not in matched_gt:
                    iou = DetectionMetrics.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                matched_pred += 1
                matched_gt.add(best_gt_idx)
        
        precision = matched_pred / (len(all_pred_boxes) + 1e-6)
        recall = matched_pred / (len(all_gt_boxes) + 1e-6)
        
        return precision, recall


class SegmentationMetrics:
    """Calculate segmentation metrics: IoU, Dice, Pixel Accuracy"""
    
    @staticmethod
    def calculate_iou(pred_mask, gt_mask, num_classes=2):
        """Calculate mean IoU"""
        ious = []
        
        for cls in range(num_classes):
            pred = (pred_mask == cls)
            gt = (gt_mask == cls)
            
            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0 if intersection == 0 else 0.0
            
            ious.append(iou)
        
        return np.mean(ious)
    
    @staticmethod
    def calculate_dice(pred_mask, gt_mask, num_classes=2):
        """Calculate Dice coefficient"""
        smooth = 1e-6
        dices = []
        
        for cls in range(num_classes):
            pred = (pred_mask == cls)
            gt = (gt_mask == cls)
            
            intersection = np.logical_and(pred, gt).sum()
            pred_sum = pred.sum()
            gt_sum = gt.sum()
            
            dice = (2 * intersection + smooth) / (pred_sum + gt_sum + smooth)
            dices.append(dice)
        
        return np.mean(dices)
    
    @staticmethod
    def calculate_pixel_accuracy(pred_mask, gt_mask):
        """Calculate pixel accuracy"""
        correct = (pred_mask == gt_mask).sum()
        total = pred_mask.size
        return correct / total
    
    @staticmethod
    def calculate_all_metrics(pred_mask, gt_mask, num_classes=2):
        """Calculate all segmentation metrics"""
        return {
            'iou': SegmentationMetrics.calculate_iou(pred_mask, gt_mask, num_classes),
            'dice': SegmentationMetrics.calculate_dice(pred_mask, gt_mask, num_classes),
            'pixel_accuracy': SegmentationMetrics.calculate_pixel_accuracy(pred_mask, gt_mask)
        }


class EvaluationLogger:
    """Log evaluation results for report"""
    
    def __init__(self, log_dir='./logs/evaluation'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.results = {
            'detection': [],
            'segmentation': []
        }
    
    def add_detection_result(self, name, map_score, precision, recall, **kwargs):
        """Add detection evaluation result"""
        self.results['detection'].append({
            'name': name,
            'mAP': map_score,
            'Precision': precision,
            'Recall': recall,
            **kwargs
        })
    
    def add_segmentation_result(self, name, iou, dice, pixel_accuracy, **kwargs):
        """Add segmentation evaluation result"""
        self.results['segmentation'].append({
            'name': name,
            'IoU': iou,
            'Dice': dice,
            'Pixel_Accuracy': pixel_accuracy,
            **kwargs
        })
    
    def save_to_csv(self):
        """Save results to CSV"""
        det_df = pd.DataFrame(self.results['detection'])
        seg_df = pd.DataFrame(self.results['segmentation'])
        
        det_df.to_csv(os.path.join(self.log_dir, 'detection_results.csv'), index=False)
        seg_df.to_csv(os.path.join(self.log_dir, 'segmentation_results.csv'), index=False)
        print(f"Saved results to {self.log_dir}")
    
    def get_comparison_table(self):
        """Get comparison table for baseline vs improved"""
        det_df = pd.DataFrame(self.results['detection'])
        seg_df = pd.DataFrame(self.results['segmentation'])
        
        if len(det_df) >= 2:
            baseline_det = det_df.iloc[0]
            improved_det = det_df.iloc[-1]
            
            det_comparison = pd.DataFrame({
                'Metric': ['mAP', 'Precision', 'Recall'],
                'Baseline': [baseline_det['mAP'], baseline_det['Precision'], baseline_det['Recall']],
                'Improved': [improved_det['mAP'], improved_det['Precision'], improved_det['Recall']],
                'Improvement': [
                    improved_det['mAP'] - baseline_det['mAP'],
                    improved_det['Precision'] - baseline_det['Precision'],
                    improved_det['Recall'] - baseline_det['Recall']
                ]
            })
        else:
            det_comparison = det_df
        
        if len(seg_df) >= 2:
            baseline_seg = seg_df.iloc[0]
            improved_seg = seg_df.iloc[-1]
            
            seg_comparison = pd.DataFrame({
                'Metric': ['IoU', 'Dice', 'Pixel Accuracy'],
                'Baseline': [baseline_seg['IoU'], baseline_seg['Dice'], baseline_seg['Pixel_Accuracy']],
                'Improved': [improved_seg['IoU'], improved_seg['Dice'], improved_seg['Pixel_Accuracy']],
                'Improvement': [
                    improved_seg['IoU'] - baseline_seg['IoU'],
                    improved_seg['Dice'] - baseline_seg['Dice'],
                    improved_seg['Pixel_Accuracy'] - baseline_seg['Pixel_Accuracy']
                ]
            })
        else:
            seg_comparison = seg_df
        
        return det_comparison, seg_comparison