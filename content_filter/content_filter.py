"""
NSFW Content Filter using multiple detection models.
Implements majority voting across 3 models to reduce false positives.
"""

import os
import cv2
import numpy as np
import requests
import inspect
from typing import Optional, Tuple, Dict, Any
import onnxruntime as ort

# Handle both relative and absolute imports
try:
    from .hash_helper import create_hash, validate_module_integrity
except ImportError:
    from hash_helper import create_hash, validate_module_integrity

# Model configurations
MODEL_CONFIGS = {
    'nsfw_1': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_1.onnx',
        'hash_url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_1.hash',
        'size': (640, 640),
        'mean': (0.0, 0.0, 0.0),
        'std': (1.0, 1.0, 1.0),
        'threshold': 0.2,
        'threshold_index': 4  # Max of columns 4+ (detection scores)
    },
    'nsfw_2': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_2.onnx',
        'hash_url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_2.hash',
        'size': (384, 384),
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'threshold': 0.25,
        'threshold_type': 'difference'  # Score difference between classes
    },
    'nsfw_3': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_3.onnx',
        'hash_url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_3.hash',
        'size': (448, 448),
        'mean': (0.48145466, 0.4578275, 0.40821073),
        'std': (0.26862954, 0.26130258, 0.27577711),
        'threshold': 10.5,
        'threshold_type': 'sum_difference'  # Sum of NSFW classes - sum of safe classes
    }
}

# Expected hash for this module (to prevent tampering)
# This will be set during initialization
MODULE_HASH = None


class ContentFilter:
    """NSFW content detection using multiple models with majority voting."""
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize content filter.
        
        Args:
            models_dir: Directory to store/load models. If None, uses default.
        """
        if models_dir is None:
            # Use custom_nodes/Facefusion_comfyui/models/content_filter
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            models_dir = os.path.join(parent_dir, 'models', 'content_filter')
        
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self._load_models()
    
    def _validate_integrity(self) -> bool:
        """Validate module hasn't been tampered with."""
        global MODULE_HASH
        if MODULE_HASH is None:
            # First run - compute and store hash
            module_path = __file__
            with open(module_path, 'rb') as f:
                MODULE_HASH = create_hash(f.read())
            return True
        
        # Validate against stored hash
        module_path = __file__
        with open(module_path, 'rb') as f:
            current_hash = create_hash(f.read())
        
        return current_hash == MODULE_HASH
    
    def _download_model(self, model_name: str) -> bool:
        """Download model and its hash file."""
        config = MODEL_CONFIGS[model_name]
        model_path = os.path.join(self.models_dir, f'{model_name}.onnx')
        hash_path = os.path.join(self.models_dir, f'{model_name}.hash')
        
        try:
            # Download hash file first
            if not os.path.exists(hash_path):
                print(f"[ContentFilter] Downloading {model_name} hash...")
                response = requests.get(config['hash_url'], timeout=30)
                response.raise_for_status()
                with open(hash_path, 'w') as f:
                    f.write(response.text.strip())
            
            # Download model if needed
            if not os.path.exists(model_path):
                print(f"[ContentFilter] Downloading {model_name} model...")
                response = requests.get(config['url'], timeout=120, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"[ContentFilter] {model_name} downloaded successfully")
            
            # Validate hash
            with open(hash_path, 'r') as f:
                expected_hash = f.read().strip()
            
            with open(model_path, 'rb') as f:
                actual_hash = create_hash(f.read())
            
            if actual_hash != expected_hash:
                print(f"[ContentFilter] Hash mismatch for {model_name}!")
                print(f"  Expected: {expected_hash}")
                print(f"  Got: {actual_hash}")
                return False
            
            return True
            
        except Exception as e:
            print(f"[ContentFilter] Error downloading {model_name}: {e}")
            return False
    
    def _load_models(self) -> None:
        """Load all NSFW detection models."""
        # Validate module integrity first
        if not self._validate_integrity():
            print("[ContentFilter] WARNING: Module integrity check failed!")
            print("[ContentFilter] Content filter may have been tampered with.")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        for model_name in MODEL_CONFIGS.keys():
            model_path = os.path.join(self.models_dir, f'{model_name}.onnx')
            
            # Download if needed
            if not os.path.exists(model_path):
                print(f"[ContentFilter] Model {model_name} not found, downloading...")
                if not self._download_model(model_name):
                    print(f"[ContentFilter] Failed to download {model_name}")
                    continue
            
            # Validate hash before loading
            hash_path = os.path.join(self.models_dir, f'{model_name}.hash')
            if os.path.exists(hash_path):
                try:
                    with open(hash_path, 'r') as f:
                        expected_hash = f.read().strip()
                    with open(model_path, 'rb') as f:
                        actual_hash = create_hash(f.read())
                    
                    if actual_hash != expected_hash:
                        print(f"[ContentFilter] Hash validation failed for {model_name}, redownloading...")
                        os.remove(model_path)
                        if not self._download_model(model_name):
                            print(f"[ContentFilter] Failed to redownload {model_name}")
                            continue
                except Exception as e:
                    print(f"[ContentFilter] Hash validation error for {model_name}: {e}")
            
            # Load ONNX session
            try:
                session = ort.InferenceSession(
                    model_path,
                    providers=providers
                )
                self.sessions[model_name] = session
                print(f"[ContentFilter] ✓ Loaded {model_name}")
            except Exception as e:
                print(f"[ContentFilter] Error loading {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[ContentFilter] Loaded {len(self.sessions)}/3 models")
    
    def _preprocess(self, frame: np.ndarray, model_name: str) -> np.ndarray:
        """Preprocess frame for specific model."""
        config = MODEL_CONFIGS[model_name]
        size = config['size']
        mean = np.array(config['mean'], dtype=np.float32)
        std = np.array(config['std'], dtype=np.float32)
        
        # Resize frame (maintaining aspect ratio, centered)
        h, w = frame.shape[:2]
        scale = min(size[0] / w, size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad to target size
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        y_offset = (size[1] - new_h) // 2
        x_offset = (size[0] - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Normalize
        normalized = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR to RGB
        normalized = (normalized - mean) / std
        
        # Transpose to CHW format and add batch dimension
        normalized = np.transpose(normalized, (2, 0, 1))
        normalized = np.expand_dims(normalized, 0)
        
        return normalized.astype(np.float32)
    
    def _detect_nsfw_1(self, frame: np.ndarray) -> bool:
        """Detect NSFW using model 1."""
        if 'nsfw_1' not in self.sessions:
            return False
        
        try:
            preprocessed = self._preprocess(frame, 'nsfw_1')
            detection = self.sessions['nsfw_1'].run(None, {'input': preprocessed})[0]
            
            # Max score from detection columns 4+
            detection_score = np.max(np.amax(detection[:, 4:], axis=1))
            threshold = MODEL_CONFIGS['nsfw_1']['threshold']
            
            return bool(detection_score > threshold)
        except Exception as e:
            print(f"[ContentFilter] Error in nsfw_1: {e}")
            return False
    
    def _detect_nsfw_2(self, frame: np.ndarray) -> bool:
        """Detect NSFW using model 2."""
        if 'nsfw_2' not in self.sessions:
            return False
        
        try:
            preprocessed = self._preprocess(frame, 'nsfw_2')
            detection = self.sessions['nsfw_2'].run(None, {'input': preprocessed})[0]
            
            # Get first element (batch output)
            if len(detection.shape) > 1:
                detection = detection[0]
            
            # Score difference between classes
            detection_score = detection[0] - detection[1]
            threshold = MODEL_CONFIGS['nsfw_2']['threshold']
            
            return bool(detection_score > threshold)
        except Exception as e:
            print(f"[ContentFilter] Error in nsfw_2: {e}")
            return False
    
    def _detect_nsfw_3(self, frame: np.ndarray) -> bool:
        """Detect NSFW using model 3."""
        if 'nsfw_3' not in self.sessions:
            return False
        
        try:
            preprocessed = self._preprocess(frame, 'nsfw_3')
            detection = self.sessions['nsfw_3'].run(None, {'input': preprocessed})[0]
            
            # Get first element (batch output)
            if len(detection.shape) > 1:
                detection = detection[0]
            
            # Sum of NSFW classes - sum of safe classes
            detection_score = (detection[2] + detection[3]) - (detection[0] + detection[1])
            threshold = MODEL_CONFIGS['nsfw_3']['threshold']
            
            return bool(detection_score > threshold)
        except Exception as e:
            print(f"[ContentFilter] Error in nsfw_3: {e}")
            return False
    
    def analyse_frame(self, frame: np.ndarray) -> bool:
        """
        Analyse frame for NSFW content using majority voting.
        
        Args:
            frame: BGR image (OpenCV format)
            
        Returns:
            True if NSFW detected by at least 2 models
        """
        # Need at least 2 models loaded
        if len(self.sessions) < 2:
            # Only print warning once
            if not hasattr(self, '_warned_once'):
                print(f"[ContentFilter] WARNING: Only {len(self.sessions)}/3 models loaded - filter disabled!")
                print("[ContentFilter] Please check model download logs above")
                self._warned_once = True
            return False
        
        is_nsfw_1 = self._detect_nsfw_1(frame)
        is_nsfw_2 = self._detect_nsfw_2(frame)
        is_nsfw_3 = self._detect_nsfw_3(frame)
        
        # Majority voting: at least 2 models must agree
        result = (is_nsfw_1 and is_nsfw_2) or (is_nsfw_1 and is_nsfw_3) or (is_nsfw_2 and is_nsfw_3)
        
        return result


# Global filter instance
_filter_instance: Optional[ContentFilter] = None


def get_filter() -> ContentFilter:
    """Get or create global filter instance."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = ContentFilter()
    return _filter_instance


def analyse_frame(frame: np.ndarray) -> bool:
    """
    Analyse frame for NSFW content.
    
    Args:
        frame: BGR image (OpenCV format)
        
    Returns:
        True if NSFW detected
    """
    filter_instance = get_filter()
    return filter_instance.analyse_frame(frame)


def blur_frame(frame: np.ndarray, blur_amount: int = 99) -> np.ndarray:
    """
    Apply heavy blur to frame for NSFW content.
    
    Args:
        frame: BGR image (OpenCV format)
        blur_amount: Kernel size for Gaussian blur (must be odd)
        
    Returns:
        Heavily blurred frame
    """
    if blur_amount % 2 == 0:
        blur_amount += 1  # Ensure odd number
    
    # Apply strong Gaussian blur
    blurred = cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)
    
    return blurred

