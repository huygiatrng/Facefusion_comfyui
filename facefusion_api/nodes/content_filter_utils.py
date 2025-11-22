"""
Content filter utilities for NSFW detection.
"""
import sys
import os

# Import content filter for NSFW detection
_content_filter_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'content_filter')
if _content_filter_path not in sys.path:
    sys.path.insert(0, _content_filter_path)

try:
    from content_filter import analyse_frame, blur_frame
    CONTENT_FILTER_AVAILABLE = True
except Exception as e:
    print(f"[FaceFusion] Content filter not available: {e}")
    CONTENT_FILTER_AVAILABLE = False
    # Define fallback functions
    def analyse_frame(frame):
        return False
    def blur_frame(frame, blur_amount=99):
        return frame

__all__ = [
    'analyse_frame',
    'blur_frame',
    'CONTENT_FILTER_AVAILABLE',
]

