"""
Face detection - Backward compatibility wrapper.
This file now imports from the reorganized detection package.
"""
from .detection import detect_faces, select_faces, get_face_detector

__all__ = [
    'detect_faces',
    'select_faces',
    'get_face_detector',
]
