"""
Local face swapping implementation - Backward compatibility wrapper.
This file now imports from the reorganized models package.
"""
# Import everything from the new structure
from .models import (
    MODEL_URLS,
    MODEL_CONFIGS,
    WARP_TEMPLATES,
    LocalFaceSwapper,
    get_local_swapper,
    FaceOccluder,
    get_face_occluder,
    FaceParser,
    get_face_parser,
)
from .swap_local import swap_faces_local

# Export everything for backward compatibility
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
    'swap_faces_local',
]
