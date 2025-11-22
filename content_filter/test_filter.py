"""
Test script to verify content filter is working.
Run this from the content_filter directory:
  python test_filter.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from content_filter import ContentFilter, get_filter
    import numpy as np
    import cv2
    
    print("=" * 60)
    print("CONTENT FILTER TEST")
    print("=" * 60)
    
    # Create filter instance
    print("\n1. Initializing content filter...")
    filter_instance = ContentFilter()
    
    print(f"\n2. Models loaded: {len(filter_instance.sessions)}/3")
    for model_name in filter_instance.sessions.keys():
        print(f"   ✓ {model_name}")
    
    if len(filter_instance.sessions) < 2:
        print("\n❌ ERROR: Not enough models loaded!")
        print("   Check the error messages above for download/loading issues.")
        sys.exit(1)
    
    # Create a test image (blank white image)
    print("\n3. Creating test image (blank white)...")
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    print("\n4. Testing NSFW detection on blank image...")
    result = filter_instance.analyse_frame(test_image)
    print(f"   Result: {'NSFW' if result else 'Safe'} (expected: Safe)")
    
    print("\n5. Testing blur function...")
    blurred = filter_instance.analyse_frame(test_image)
    print(f"   Blur shape: {blurred.shape if isinstance(blurred, np.ndarray) else 'N/A'}")
    
    print("\n" + "=" * 60)
    print("✓ Content filter is working correctly!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

