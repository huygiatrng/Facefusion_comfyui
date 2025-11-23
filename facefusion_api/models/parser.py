"""
Face parser (bisenet) for segmenting face regions.
"""
import os
from typing import Optional, List

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from ..utils import VisionFrame, get_model_path, ensure_model_exists
from .constants import MODEL_URLS, MODEL_CONFIGS


class FaceParser:
    """Face parser (bisenet) for segmenting face regions."""
    
    def __init__(self, model_name: str = 'bisenet_resnet_34'):
        self.model_name = model_name
        self.model_session = None
        self.model_config = MODEL_CONFIGS.get(model_name, {'size': (512, 512), 'type': 'parser'})
    
    def initialize(self) -> bool:
        """Initialize the parser model."""
        try:
            model_path = get_model_path(f'{self.model_name}.onnx')
            
            if not os.path.exists(model_path):
                print(f"[FaceParser] Downloading model: {self.model_name}")
                download_url = MODEL_URLS.get(self.model_name)
                if not download_url:
                    print(f"[FaceParser] Error: Unknown model {self.model_name}")
                    return False
                    
                if not ensure_model_exists(f'{self.model_name}.onnx', download_url):
                    print(f"[FaceParser] Error: Failed to download model")
                    return False
            
            # Create ONNX session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.model_session = ort.InferenceSession(model_path, providers=providers)
            
            # print(f"[FaceParser] Model {self.model_name} loaded, running on: {self.model_session.get_providers()[0]}")
            return True
        except Exception as e:
            print(f"[FaceParser] Error initializing model: {e}")
            return False
    
    def create_region_mask(self, crop_frame: VisionFrame, include_regions: List[int] = None) -> Optional[NDArray]:
        """Create mask from face regions. 
        
        Common regions:
        - 1: face skin
        - 2-3: eyebrows
        - 4-5: eyes
        - 6: nose
        - 7-9: mouth/lips
        - 10-13: ears
        - 14-16: glasses/accessories
        - 17: hair
        - 18: hat
        - 19: earring/earpiece
        
        By default, includes face skin and facial features (1-13)
        """
        if self.model_session is None:
            if not self.initialize():
                return None
        
        if include_regions is None:
            # Default: face skin + eyebrows + eyes + nose + mouth + ears
            include_regions = list(range(1, 14))
        
        try:
            size = self.model_config['size']
            
            # Resize to model input size
            prepared = cv2.resize(crop_frame, size)
            
            # Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet)
            prepared = prepared.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            prepared = (prepared - mean) / std
            
            # Transpose to CHW and add batch
            prepared = prepared.transpose(2, 0, 1)
            prepared = np.expand_dims(prepared, 0)
            
            # Run inference
            outputs = self.model_session.run(None, {'input': prepared})
            segmentation = outputs[0][0]  # Get first output, remove batch
            
            # Get class predictions (argmax across classes)
            if len(segmentation.shape) == 3:
                # Shape is [num_classes, H, W]
                segmentation = np.argmax(segmentation, axis=0)
            
            # Create mask from selected regions
            mask = np.zeros_like(segmentation, dtype=np.float32)
            for region_id in include_regions:
                mask[segmentation == region_id] = 1.0
            
            # Resize mask back to crop frame size
            mask = cv2.resize(mask, (crop_frame.shape[1], crop_frame.shape[0]))
            
            return mask
            
        except Exception as e:
            print(f"[FaceParser] Error creating mask: {e}")
            return None


# Global instances
_parser_instances = {}


def get_face_parser(model_name: str = 'bisenet_resnet_34') -> Optional[FaceParser]:
    """Get or create face parser instance."""
    global _parser_instances
    if model_name not in _parser_instances:
        _parser_instances[model_name] = FaceParser(model_name)
    return _parser_instances[model_name]







