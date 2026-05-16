"""
SmolVLM integration for Visual Question Answering with UNet segmentation context
"""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import cv2
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path

from .config import vqa_config


class SmolVLMEngine:
    """SmolVLM model for question answering about fabric defects with UNet context"""
    
    def __init__(self, device: str = None, segmentation_model=None):
        if device is None:
            device = vqa_config.device
        self.device = torch.device(device)
        self.segmentation_model = segmentation_model
        
        print(f"🚀 Loading SmolVLM model on {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(vqa_config.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            vqa_config.model_name,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        if self.device.type == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("✅ SmolVLM loaded successfully!")
        
        if self.segmentation_model:
            print("✅ UNet segmentation model integrated for defect context")
    
    def _preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Preprocess image for SmolVLM (convert to RGB and resize)"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize to target size
        rgb = cv2.resize(rgb, target_size)
        
        return Image.fromarray(rgb)
    
    def _run_segmentation(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run UNet segmentation to detect defects
        
        Returns:
            Dictionary containing segmentation results
        """
        if self.segmentation_model is None:
            return {
                'has_defects': False,
                'num_defects': 0,
                'defects': [],
                'defect_features': {},
                'mask': None,
                'error': 'Segmentation model not available'
            }
        
        try:
            # Run segmentation prediction
            seg_result = self.segmentation_model.predict(image)
            
            return {
                'has_defects': seg_result['num_defects'] > 0,
                'num_defects': seg_result['num_defects'],
                'defects': seg_result.get('defects', []),
                'defect_features': seg_result.get('defect_features', {}),
                'mask': seg_result.get('mask'),
                'prob': seg_result.get('prob'),
                'success': True
            }
            
        except Exception as e:
            print(f"⚠️ Segmentation inference failed: {e}")
            return {
                'has_defects': False,
                'num_defects': 0,
                'defects': [],
                'defect_features': {},
                'mask': None,
                'error': str(e),
                'success': False
            }
    
    def _format_segmentation_context(self, seg_result: Dict[str, Any]) -> str:
        """
        Format UNet segmentation results into a rich context string for VLM
        
        Args:
            seg_result: Segmentation results from _run_segmentation
        
        Returns:
            Formatted context string
        """
        if not seg_result or not seg_result.get('success', False):
            return ""
        
        num_defects = seg_result['num_defects']
        defects = seg_result.get('defects', [])
        features = seg_result.get('defect_features', {})
        
        context_parts = []
        
        # Header
        context_parts.append("=== DEFECT DETECTION RESULTS (from U-Net Segmentation) ===")
        
        # Overall statistics
        if num_defects == 0:
            context_parts.append("No defects detected on this fabric.")
            return "\n".join(context_parts)
        
        context_parts.append(f"Total defects detected: {num_defects}")
        
        if features:
            context_parts.append(f"Total defect area: {features.get('total_area', 0)} pixels")
            context_parts.append(f"Average defect area: {features.get('avg_area', 0):.1f} pixels")
            context_parts.append(f"Largest defect area: {features.get('max_area', 0)} pixels")
            context_parts.append(f"Smallest defect area: {features.get('min_area', 0)} pixels")
        
        # Severity assessment
        severity = self._assess_severity(features)
        context_parts.append(f"Overall severity: {severity}")
        
        # Detailed defect list (limit to first 10 for token efficiency)
        if defects and len(defects) > 0:
            context_parts.append("\nDetailed defect information:")
            for i, defect in enumerate(defects[:10], 1):
                bbox = defect.get('bbox', (0, 0, 0, 0))
                centroid = defect.get('centroid', (0, 0))
                area = defect.get('area', 0)
                
                context_parts.append(
                    f"  Defect #{i}: area={area}px, "
                    f"position=({centroid[0]:.0f}, {centroid[1]:.0f}), "
                    f"bbox=[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                )
            
            if len(defects) > 10:
                context_parts.append(f"  ... and {len(defects) - 10} more defects")
        
        context_parts.append("=== END OF SEGMENTATION RESULTS ===\n")
        
        return "\n".join(context_parts)
    
    def _assess_severity(self, features: Dict[str, Any]) -> str:
        """Assess defect severity based on segmentation results"""
        if not features:
            return "unknown"
        
        num_defects = features.get('num_defects', 0)
        total_area = features.get('total_area', 0)
        max_area = features.get('max_area', 0)
        
        if num_defects == 0:
            return "none"
        
        # Heuristic severity assessment
        if num_defects > 20 or total_area > 50000:
            return "critical"
        elif num_defects > 10 or total_area > 20000 or max_area > 10000:
            return "high"
        elif num_defects > 5 or total_area > 5000:
            return "medium"
        else:
            return "low"

    def _generate_rule_based_answer(self, question: str, seg_result: Dict[str, Any]) -> Optional[str]:
        """
        Generate answer using rule-based logic from segmentation results.
        Returns None if question doesn't match rules (falls back to VLM).
        """
        if not seg_result or not seg_result.get('success', False):
            return None
        
        question_lower = question.lower()
        num_defects = seg_result['num_defects']
        defects = seg_result.get('defects', [])
        features = seg_result.get('defect_features', {})
        
        # === Questions about count ===
        if any(word in question_lower for word in ['how many', 'count', 'number of', 'total defects']):
            if num_defects == 0:
                return "No defects detected on this fabric."
            elif num_defects == 1:
                return f"There is 1 defect detected on this fabric."
            else:
                return f"There are {num_defects} defects detected on this fabric."
        
        # === Questions about existence ===
        if any(word in question_lower for word in ['any defect', 'is there', 'defect present', 'any issue']):
            if num_defects == 0:
                return "No, there are no defects detected on this fabric."
            else:
                return f"Yes, there {'is 1 defect' if num_defects == 1 else f'are {num_defects} defects'} detected on this fabric."
        
        # === Questions about position / location ===
        if any(word in question_lower for word in ['position', 'location', 'where', 'located', 'place', 'side', 'corner', 'top', 'bottom', 'left', 'right', 'center']):
            if num_defects == 0:
                return "No defects to locate."
            
            # Specific: largest defect position
            if 'largest' in question_lower or 'biggest' in question_lower:
                largest = max(defects, key=lambda d: d.get('area', 0))
                pos = largest.get('position', 'unknown location')
                return f"The largest defect is located at the {pos}."
            
            # Specific: smallest defect position
            if 'smallest' in question_lower:
                smallest = min(defects, key=lambda d: d.get('area', float('inf')))
                pos = smallest.get('position', 'unknown location')
                return f"The smallest defect is located at the {pos}."
            
            # Count defects by position
            if 'how many' in question_lower and ('left' in question_lower or 'right' in question_lower or 'top' in question_lower or 'bottom' in question_lower or 'center' in question_lower):
                target_pos = None
                if 'left' in question_lower:
                    target_pos = 'left'
                elif 'right' in question_lower:
                    target_pos = 'right'
                elif 'top' in question_lower:
                    target_pos = 'top'
                elif 'bottom' in question_lower:
                    target_pos = 'bottom'
                elif 'center' in question_lower or 'centre' in question_lower:
                    target_pos = 'center'
                
                if target_pos:
                    count = sum(1 for d in defects if d.get('horizontal_position') == target_pos or d.get('vertical_position') == target_pos)
                    side_name = {'left': 'left side', 'right': 'right side', 'top': 'top side', 'bottom': 'bottom side', 'center': 'center'}[target_pos]
                    if count == 0:
                        return f"There are no defects on the {side_name} of the fabric."
                    elif count == 1:
                        return f"There is 1 defect on the {side_name} of the fabric."
                    else:
                        return f"There are {count} defects on the {side_name} of the fabric."
            
            # Single defect
            if num_defects == 1:
                pos = defects[0].get('position', 'unknown location')
                return f"The only defect is located at the {pos}."
            
            # Multiple defects (list all positions, limit to first 5)
            if num_defects <= 5:
                positions = [f"Defect {i+1} is at the {d.get('position', 'unknown location')}" 
                            for i, d in enumerate(defects)]
                return "Position details: " + "; ".join(positions) + "."
            else:
                # Group by position type
                position_counts = {}
                for d in defects:
                    pos = d.get('position', 'unknown')
                    position_counts[pos] = position_counts.get(pos, 0) + 1
                
                summary = [f"{count} defect(s) at the {pos}" for pos, count in position_counts.items()]
                return "Defect distribution: " + ", ".join(summary) + "."
        
        # === Questions about size / dimensions ===
        if any(word in question_lower for word in ['size', 'dimension', 'how big', 'how large', 'width', 'height', 'mm', 'millimeter', 'measurement']):
            if num_defects == 0:
                return "No defects to measure."
            
            # Helper function to format size
            def format_size(width_mm, height_mm):
                return f"{width_mm:.1f}mm x {height_mm:.1f}mm"
            
            # Largest defect size
            if 'largest' in question_lower or 'biggest' in question_lower:
                largest = max(defects, key=lambda d: d.get('area', 0))
                pos = largest.get('position', '')
                width_mm = largest.get('width_mm', 0)
                height_mm = largest.get('height_mm', 0)
                if pos:
                    return f"The largest defect (at the {pos}) measures {format_size(width_mm, height_mm)}."
                else:
                    return f"The largest defect measures {format_size(width_mm, height_mm)}."
            
            # Smallest defect size
            if 'smallest' in question_lower:
                smallest = min(defects, key=lambda d: d.get('area', float('inf')))
                pos = smallest.get('position', '')
                width_mm = smallest.get('width_mm', 0)
                height_mm = smallest.get('height_mm', 0)
                if pos:
                    return f"The smallest defect (at the {pos}) measures {format_size(width_mm, height_mm)}."
                else:
                    return f"The smallest defect measures {format_size(width_mm, height_mm)}."
            
            # Specific defect by number (e.g., "defect 2", "second defect")
            import re
            match = re.search(r'(?:defect|#?)\s*(\d+)(?:st|nd|rd|th)?', question_lower)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(defects):
                    d = defects[idx]
                    pos = d.get('position', '')
                    width_mm = d.get('width_mm', 0)
                    height_mm = d.get('height_mm', 0)
                    if pos:
                        return f"Defect #{idx+1} (at the {pos}) measures {format_size(width_mm, height_mm)}."
                    else:
                        return f"Defect #{idx+1} measures {format_size(width_mm, height_mm)}."
            
            # Single defect
            if num_defects == 1:
                d = defects[0]
                pos = d.get('position', '')
                width_mm = d.get('width_mm', 0)
                height_mm = d.get('height_mm', 0)
                if pos:
                    return f"The defect at the {pos} measures {format_size(width_mm, height_mm)}."
                else:
                    return f"The defect measures {format_size(width_mm, height_mm)}."
            
            # Multiple defects - list sizes for first few
            descriptions = []
            for i, d in enumerate(defects[:5]):
                pos = d.get('position', '')
                width_mm = d.get('width_mm', 0)
                height_mm = d.get('height_mm', 0)
                if pos:
                    descriptions.append(f"Defect {i+1} (at the {pos}): {format_size(width_mm, height_mm)}")
                else:
                    descriptions.append(f"Defect {i+1}: {format_size(width_mm, height_mm)}")
            
            if len(defects) > 5:
                descriptions.append(f"... and {len(defects) - 5} more defects")
            
            return "Defect measurements:\n- " + "\n- ".join(descriptions)
        
        # === Questions about area ===
        if any(word in question_lower for word in ['area', 'total area', 'coverage']):
            if num_defects == 0:
                return "No defects detected, so total area is 0 mm²."
            
            total_area_mm2 = features.get('total_area_mm2', 0)
            return f"Total defect area is {total_area_mm2:.1f} square millimeters."
        
        # === Questions about severity ===
        if any(word in question_lower for word in ['severity', 'serious', 'how bad', 'how severe']):
            severity = features.get('severity', 'none')
            severity_desc = {
                'critical': 'critical - immediate attention required',
                'high': 'high - significant defects present',
                'medium': 'medium - moderate defects detected',
                'low': 'low - minor defects only',
                'none': 'none - fabric appears defect-free'
            }
            return f"Defect severity level: {severity_desc.get(severity, severity)}."
        
        # === Questions about the largest defect's area ===
        if 'largest' in question_lower and ('area' in question_lower or 'big' in question_lower):
            if num_defects == 0:
                return "No defects to measure."
            largest = max(defects, key=lambda d: d.get('area', 0))
            pos = largest.get('position', '')
            area_mm2 = largest.get('area_mm2', 0)
            if pos:
                return f"The largest defect at the {pos} has an area of {area_mm2:.1f} mm²."
            else:
                return f"The largest defect has an area of {area_mm2:.1f} mm²."
        
        # === Questions about the smallest defect's area ===
        if 'smallest' in question_lower and ('area' in question_lower or 'small' in question_lower):
            if num_defects == 0:
                return "No defects to measure."
            smallest = min(defects, key=lambda d: d.get('area', float('inf')))
            pos = smallest.get('position', '')
            area_mm2 = smallest.get('area_mm2', 0)
            if pos:
                return f"The smallest defect at the {pos} has an area of {area_mm2:.1f} mm²."
            else:
                return f"The smallest defect has an area of {area_mm2:.1f} mm²."
        
        return None  # Fall back to VLM for complex questions
    
    def answer(
        self, 
        image: np.ndarray, 
        question: str, 
        use_segmentation: bool = True,
        use_rule_based: bool = True,
        custom_context: Optional[str] = None
    ) -> str:
        """
        Answer a question about an image with optional UNet segmentation context
        
        Args:
            image: Image as numpy array (BGR format)
            question: Question text
            use_segmentation: Whether to include UNet segmentation context
            use_rule_based: Whether to try rule-based answers first
            custom_context: Optional additional context
        
        Returns:
            Answer string
        """
        seg_result = None
        rule_answer = None
        
        # Run segmentation if requested
        if use_segmentation and self.segmentation_model:
            seg_result = self._run_segmentation(image)
            
            # Try rule-based answer first
            if use_rule_based:
                rule_answer = self._generate_rule_based_answer(question, seg_result)
                if rule_answer:
                    return rule_answer
        
        # Build context for VLM
        context_parts = []
        
        if custom_context:
            context_parts.append(custom_context)
        
        if use_segmentation and seg_result and seg_result.get('success', False):
            context_parts.append(self._format_segmentation_context(seg_result))
        
        # Prepare image for VLM
        pil_image = self._preprocess_image(image)
        
        # Build enhanced question with context
        if context_parts:
            enhanced_question = f"""{chr(10).join(context_parts)}

Based on the analysis results above, please answer the following question about the fabric defect image:

Question: {question}

Please provide a clear and specific answer based on the detected defects."""
        else:
            enhanced_question = question
        
        # Format chat for SmolVLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": enhanced_question}
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[pil_image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=vqa_config.max_new_tokens,
                do_sample=False,
                temperature=vqa_config.temperature
            )
        
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "assistant" in answer:
            answer = answer.split("assistant")[-1].strip()
        if answer.startswith("\n"):
            answer = answer[1:]
        
        return answer
    
    def answer_with_details(
        self, 
        image: np.ndarray, 
        question: str,
        use_segmentation: bool = True
    ) -> Dict[str, Any]:
        """
        Answer question and return both answer and segmentation details
        
        Returns:
            Dictionary with 'answer', 'segmentation_result', and 'context_used'
        """
        seg_result = None
        context = None
        
        if use_segmentation and self.segmentation_model:
            seg_result = self._run_segmentation(image)
            context = self._format_segmentation_context(seg_result)
            
            # Try rule-based first
            rule_answer = self._generate_rule_based_answer(question, seg_result)
            if rule_answer:
                return {
                    'answer': rule_answer,
                    'segmentation_result': seg_result,
                    'context_used': context,
                    'method': 'rule_based'
                }
        
        # Fall back to VLM
        answer = self.answer(image, question, use_segmentation=use_segmentation)
        
        return {
            'answer': answer,
            'segmentation_result': seg_result,
            'context_used': context,
            'method': 'vlm'
        }
    
    def batch_answer(
        self,
        images: List[np.ndarray],
        questions: List[str],
        use_segmentation: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions for multiple images
        
        Args:
            images: List of images
            questions: List of questions (same length as images)
            use_segmentation: Whether to use segmentation context
        
        Returns:
            List of results
        """
        if len(images) != len(questions):
            raise ValueError(f"Number of images ({len(images)}) != number of questions ({len(questions)})")
        
        results = []
        for img, q in zip(images, questions):
            results.append(self.answer_with_details(img, q, use_segmentation))
        
        return results
    
    def get_segmentation_only(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Get segmentation results without answering a question
        
        Args:
            image: Image as numpy array
        
        Returns:
            Segmentation results dictionary
        """
        if self.segmentation_model:
            return self._run_segmentation(image)
        return {'error': 'Segmentation model not available', 'success': False}