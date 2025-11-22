"""
Face occluder (xseg) for detecting occluded regions.
"""
import os
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from ..utils import VisionFrame, get_model_path, ensure_model_exists
from .constants import MODEL_URLS, MODEL_CONFIGS


class FaceOccluder:
    """Face occluder (xseg) for detecting occluded regions."""
    
    def __init__(self, model_name: str = 'xseg_1'):
        self.model_name = model_name
        self.model_session = None
        self.model_config = MODEL_CONFIGS.get(model_name, {'size': (256, 256), 'type': 'occluder'})
    
    def initialize(self) -> bool:
        """Initialize the occluder model."""
        try:
            model_path = get_model_path(f'{self.model_name}.onnx')
            
            if not os.path.exists(model_path):
                print(f"[FaceOccluder] Downloading model: {self.model_name}")
                download_url = MODEL_URLS.get(self.model_name)
                if not download_url:
                    print(f"[FaceOccluder] Error: Unknown model {self.model_name}")
                    return False
                    
                if not ensure_model_exists(f'{self.model_name}.onnx', download_url):
                    print(f"[FaceOccluder] Error: Failed to download model")
                    return False
            
            # Create ONNX session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.model_session = ort.InferenceSession(model_path, providers=providers)
            
            # print(f"[FaceOccluder] Model {self.model_name} loaded, running on: {self.model_session.get_providers()[0]}")
            return True
        except Exception as e:
            print(f"[FaceOccluder] Error initializing model: {e}")
            return False
    
    def create_occlusion_mask(self, crop_frame: VisionFrame) -> Optional[NDArray]:
        """Create occlusion mask for the cropped face."""
        if self.model_session is None:
            if not self.initialize():
                return None
        
        try:
            size = self.model_config['size']
            
            # Resize to model input size
            prepared = cv2.resize(crop_frame, size)
            
            # Normalize to [0, 1] - keep HWC format for xseg models
            prepared = prepared.astype(np.float32) / 255.0
            prepared = np.expand_dims(prepared, 0)  # Add batch dimension: (1, H, W, C)
            
            # Run inference
            outputs = self.model_session.run(None, {'input': prepared})
            mask = outputs[0][0]  # Get first output, remove batch
            
            # Process mask - xseg outputs a single channel mask
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]  # Take first channel if multiple (HWC format)
            
            # Resize mask back to crop frame size
            mask = cv2.resize(mask, (crop_frame.shape[1], crop_frame.shape[0]))
            
            # Threshold and invert - xseg outputs 1 for face, 0 for occlusion
            # We want 1 for visible, 0 for occluded
            mask = np.clip(mask, 0, 1)
            
            return mask
            
        except Exception as e:
            print(f"[FaceOccluder] Error creating mask: {e}")
            return None


# Global instances
_occluder_instances = {}


def get_face_occluder(model_name: str = 'xseg_1') -> Optional[FaceOccluder]:
    """Get or create face occluder instance."""
    global _occluder_instances
    if model_name not in _occluder_instances:
        _occluder_instances[model_name] = FaceOccluder(model_name)
    return _occluder_instances[model_name]




