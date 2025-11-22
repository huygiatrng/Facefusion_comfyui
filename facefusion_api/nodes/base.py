"""
Base imports and utilities for all ComfyUI nodes.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import Tuple, Optional, List, Dict, Any

import torch
import numpy as np
from comfy.comfy_types import IO
from comfy_api.input_impl.video_types import VideoFromComponents
from comfy_api.util import VideoComponents
from comfy_api_nodes.util import bytesio_to_image_tensor, tensor_to_bytesio
from httpx import Client as HttpClient, Headers
from httpx_retries import Retry, RetryTransport
from torch import Tensor

from ..types import FaceSwapperModel, InputTypes
from ..utils import tensor_to_cv2, cv2_to_tensor, get_average_embedding, implode_pixel_boost, explode_pixel_boost
from ..detection import detect_faces, select_faces
from ..swap_local import swap_faces_local
from ..models import get_local_swapper, get_face_occluder, get_face_parser, MODEL_CONFIGS
from .content_filter_utils import analyse_frame, blur_frame, CONTENT_FILTER_AVAILABLE

__all__ = [
    # Standard library
    'ThreadPoolExecutor',
    'partial',
    'BytesIO',
    'Tuple',
    'Optional',
    'List',
    'Dict',
    'Any',
    # PyTorch
    'torch',
    'Tensor',
    'np',
    # ComfyUI
    'IO',
    'VideoFromComponents',
    'VideoComponents',
    'bytesio_to_image_tensor',
    'tensor_to_bytesio',
    'HttpClient',
    'Headers',
    'Retry',
    'RetryTransport',
    # Our types
    'FaceSwapperModel',
    'InputTypes',
    # Our utilities
    'tensor_to_cv2',
    'cv2_to_tensor',
    'get_average_embedding',
    'implode_pixel_boost',
    'explode_pixel_boost',
    'detect_faces',
    'select_faces',
    'swap_faces_local',
    'get_local_swapper',
    'get_face_occluder',
    'get_face_parser',
    'MODEL_CONFIGS',
    'analyse_frame',
    'blur_frame',
    'CONTENT_FILTER_AVAILABLE',
]





