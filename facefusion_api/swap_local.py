"""
Local face swapping main function.
"""
from typing import Optional, List, Tuple

from .utils import VisionFrame
from .detection.detector import detect_faces
from .models import get_local_swapper, get_face_occluder, get_face_parser


def swap_faces_local(
    source_image: VisionFrame,
    target_image: VisionFrame,
    model_name: str = 'hyperswap_1c_256',
    pixel_boost: str = '512x512',
    face_mask_blur: float = 0.3,
    face_selector_mode: str = 'one',
    face_position: int = 0,
    sort_order: str = 'large-small',
    score_threshold: float = 0.3,
    face_occluder_model: Optional[str] = None,
    face_parser_model: Optional[str] = None,
    face_detector_model: str = 'scrfd',
    face_mask_types: Optional[List[str]] = None,
    face_mask_areas: Optional[List[str]] = None,
    face_mask_regions: Optional[List[str]] = None,
    face_mask_padding: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> VisionFrame:
    """
    Swap faces locally using ONNX models.
    
    Args:
        source_image: Source face image
        target_image: Target image to swap faces in
        model_name: Face swapper model to use
        pixel_boost: Resolution for pixel boost (256x256, 512x512, 768x768, 1024x1024)
        face_mask_blur: Blur amount for mask blending (0.0-1.0, default 0.3)
        face_selector_mode: How to select faces ('one', 'many')
        face_position: Which face to use from source
        sort_order: How to sort detected faces (large-small, left-right, etc.)
        score_threshold: Minimum detection confidence
        face_occluder_model: Face occluder model to use (xseg_1, xseg_2, xseg_3) for masking occlusions
        face_parser_model: Face parser model to use (bisenet_resnet_18, bisenet_resnet_34) for region segmentation
        face_detector_model: Face detector model to use (scrfd, retinaface, yolo_face, yunet, many)
        face_mask_types: List of mask types to use ['box', 'occlusion', 'area', 'region']
        face_mask_areas: List of face areas for area mask ['upper-face', 'lower-face', 'mouth']
        face_mask_regions: List of face regions for region mask ['skin', 'nose', 'mouth', etc.]
        face_mask_padding: Padding for box mask (top, right, bottom, left)
    
    Returns:
        Image with swapped faces
    """
    # Default mask types if not specified
    if face_mask_types is None:
        face_mask_types = ['box']
    # print(f"[LocalSwap] Starting local face swap with model: {model_name}")
    
    # Detect faces in source and target
    source_faces = detect_faces(source_image, score_threshold, sort_order, face_detector_model)
    target_faces = detect_faces(target_image, score_threshold, sort_order, face_detector_model)
    
    if not source_faces:
        # print("[LocalSwap] No faces detected in source image")
        return target_image
    
    if not target_faces:
        # print("[LocalSwap] No faces detected in target image")
        return target_image
    
    # Select source face
    source_face = source_faces[min(face_position, len(source_faces) - 1)]
    # print(f"[LocalSwap] Using source face at position {face_position}")
    
    # Get swapper
    swapper = get_local_swapper(model_name)
    
    # Get occluder and parser if specified
    occluder = None
    parser = None
    if face_occluder_model and face_occluder_model != 'none':
        occluder = get_face_occluder(face_occluder_model)
    if face_parser_model and face_parser_model != 'none':
        parser = get_face_parser(face_parser_model)
    
    # Swap faces - pass occluder and parser instances
    result = target_image.copy()
    
    if face_selector_mode == 'many':
        # Swap all faces
        for i, target_face in enumerate(target_faces):
            # print(f"[LocalSwap] Swapping face {i+1}/{len(target_faces)}")
            result = swapper.swap_face(
                source_face, target_face, result, pixel_boost, face_mask_blur, 
                occluder, parser, source_image,
                face_mask_types, face_mask_areas, face_mask_regions, face_mask_padding
            )
    else:
        # Swap one face
        target_face = target_faces[min(face_position, len(target_faces) - 1)]
        # print(f"[LocalSwap] Swapping target face at position {face_position}")
        result = swapper.swap_face(
            source_face, target_face, result, pixel_boost, face_mask_blur,
            occluder, parser, source_image,
            face_mask_types, face_mask_areas, face_mask_regions, face_mask_padding
        )
    
    # print("[LocalSwap] Face swap completed")
    return result

