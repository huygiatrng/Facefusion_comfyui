"""
Face parser (bisenet) for segmenting face regions.
"""
import os
from typing import Optional, List, Union

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from ..utils import VisionFrame, get_model_path, ensure_model_exists
from .constants import MODEL_URLS, MODEL_CONFIGS, FACE_MASK_REGION_SET


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
    
    def create_region_mask(self, crop_frame: VisionFrame, include_regions: Union[List[int], List[str]] = None) -> Optional[NDArray]:
        """Create mask from face regions (matching facefusion implementation).
        
        Region names (strings) are converted to IDs using FACE_MASK_REGION_SET:
        - 'skin': 1
        - 'left-eyebrow': 2
        - 'right-eyebrow': 3
        - 'left-eye': 4
        - 'right-eye': 5
        - 'glasses': 6
        - 'nose': 10
        - 'mouth': 11
        - 'upper-lip': 12
        - 'lower-lip': 13
        
        Can also pass region IDs (integers) directly.
        By default, includes face skin and facial features.
        """
        if self.model_session is None:
            if not self.initialize():
                return None
        
        # Convert region names to IDs if strings are provided
        region_ids = []
        if include_regions is None:
            # Default: face skin + eyebrows + eyes + nose + mouth
            region_ids = list(FACE_MASK_REGION_SET.values())
        else:
            for region in include_regions:
                if isinstance(region, str):
                    if region in FACE_MASK_REGION_SET:
                        region_ids.append(FACE_MASK_REGION_SET[region])
                elif isinstance(region, int):
                    region_ids.append(region)
        
        if not region_ids:
            region_ids = list(FACE_MASK_REGION_SET.values())
        
        try:
            size = self.model_config['size']
            
            # Resize to model input size
            prepared = cv2.resize(crop_frame, size)
            
            # Convert BGR to RGB and normalize with ImageNet mean/std (matching facefusion)
            prepared = prepared[:, :, ::-1].astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            prepared = (prepared - mean) / std
            
            # Transpose to CHW and add batch
            prepared = prepared.transpose(2, 0, 1)
            prepared = np.expand_dims(prepared, 0)
            
            # Run inference
            outputs = self.model_session.run(None, {'input': prepared})
            segmentation = outputs[0][0]  # Get first output, remove batch
            
            # Create mask by checking if argmax is in region_ids (matching facefusion)
            region_mask = np.isin(segmentation.argmax(0), region_ids)
            
            # Resize mask back to crop frame size
            region_mask = cv2.resize(region_mask.astype(np.float32), (crop_frame.shape[1], crop_frame.shape[0]))
            
            # Apply Gaussian blur for smooth edges (matching facefusion)
            region_mask = (cv2.GaussianBlur(region_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
            
            return region_mask
            
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













