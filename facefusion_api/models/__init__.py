"""
Face swapping models package.
"""
from .constants import MODEL_URLS, MODEL_CONFIGS, WARP_TEMPLATES
from .swapper import LocalFaceSwapper, get_local_swapper
from .occluder import FaceOccluder, get_face_occluder
from .parser import FaceParser, get_face_parser

__all__ = [
    'MODEL_URLS',
    'MODEL_CONFIGS',
    'WARP_TEMPLATES',
    'LocalFaceSwapper',
    'get_local_swapper',
    'FaceOccluder',
    'get_face_occluder',
    'FaceParser',
    'get_face_parser',
]





