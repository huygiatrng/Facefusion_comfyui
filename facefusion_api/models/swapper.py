"""
Local face swapper using ONNX models.
"""
import os
from typing import Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from ..utils import VisionFrame, Face, get_model_path, ensure_model_exists, implode_pixel_boost, explode_pixel_boost
from .constants import MODEL_URLS, MODEL_CONFIGS, WARP_TEMPLATES

if TYPE_CHECKING:
    from .occluder import FaceOccluder
    from .parser import FaceParser


class LocalFaceSwapper:
    """Local face swapping using ONNX models."""
    
    def __init__(self, model_name: str = 'hyperswap_1c_256'):
        self.model_name = model_name
        self.model_session = None
        self.model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['hyperswap_1c_256'])
        
    def initialize(self) -> bool:
        """Initialize the face swapper model."""
        try:
            model_path = get_model_path(f'{self.model_name}.onnx')
            
            if not os.path.exists(model_path):
                print(f"[LocalFaceSwapper] Downloading model: {self.model_name}")
                download_url = MODEL_URLS.get(self.model_name)
                if not download_url:
                    print(f"[LocalFaceSwapper] Error: Unknown model {self.model_name}")
                    return False
                    
                if not ensure_model_exists(f'{self.model_name}.onnx', download_url):
                    print(f"[LocalFaceSwapper] Error: Failed to download model")
                    return False
            
            # Create ONNX session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.model_session = ort.InferenceSession(model_path, providers=providers)
            
            # print(f"[LocalFaceSwapper] Model {self.model_name} loaded successfully")
            print(f"[LocalFaceSwapper] Running on: {self.model_session.get_providers()[0]}")
            return True
        except Exception as e:
            print(f"[LocalFaceSwapper] Error initializing model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def swap_face(
        self,
        source_face: Face,
        target_face: Face,
        target_image: VisionFrame,
        pixel_boost: str = '512x512',
        face_mask_blur: float = 0.3,
        face_occluder: Optional['FaceOccluder'] = None,
        face_parser: Optional['FaceParser'] = None
    ) -> VisionFrame:
        """Swap a single face in the target image."""
        if self.model_session is None:
            if not self.initialize():
                return target_image
        
        try:
            # Parse pixel boost resolution
            pixel_boost_width, pixel_boost_height = map(int, pixel_boost.split('x'))
            pixel_boost_size = (pixel_boost_width, pixel_boost_height)
            
            # Get model configuration
            model_size = self.model_config['size']
            model_template = self.model_config['template']
            model_type = self.model_config['type']
            
            # Calculate pixel boost factor
            pixel_boost_total = pixel_boost_size[0] // model_size[0]
            
            # print(f"[LocalFaceSwapper] Swapping face with pixel_boost={pixel_boost} (factor={pixel_boost_total}x), blur={face_mask_blur}")
            
            # Warp target face to pixel boost size
            crop_frame_original, affine_matrix = self._warp_face(
                target_image,
                target_face['landmarks'],
                model_template,
                pixel_boost_size
            )
            
            # Create occluder and parser masks on ORIGINAL target face BEFORE swapping
            occluder_mask = None
            parser_mask = None
            
            if face_occluder is not None:
                # Create occluder mask on the ORIGINAL crop frame
                # print(f"[LocalFaceSwapper] Creating occluder mask with {face_occluder.model_name}")
                occluder_mask = face_occluder.create_occlusion_mask(crop_frame_original)
                # if occluder_mask is not None:
                #     print(f"[LocalFaceSwapper] Occluder mask created: min={occluder_mask.min():.3f}, max={occluder_mask.max():.3f}, mean={occluder_mask.mean():.3f}")
            
            if face_parser is not None:
                # Create parser mask on the ORIGINAL crop frame
                # print(f"[LocalFaceSwapper] Creating parser mask with {face_parser.model_name}")
                parser_mask = face_parser.create_region_mask(crop_frame_original)
                # if parser_mask is not None:
                #     print(f"[LocalFaceSwapper] Parser mask created: min={parser_mask.min():.3f}, max={parser_mask.max():.3f}, mean={parser_mask.mean():.3f}")
            
            # Split into patches for pixel boost
            if pixel_boost_total > 1:
                crop_patches = implode_pixel_boost(crop_frame_original, pixel_boost_total, model_size)
            else:
                crop_patches = [crop_frame_original]
            
            # Process each patch through the model
            swapped_patches = []
            for patch in crop_patches:
                # Prepare inputs
                patch_prepared = self._prepare_crop_frame(patch)
                source_embedding = source_face.get('embedding')
                
                if source_embedding is None:
                    print("[LocalFaceSwapper] Warning: No source embedding, using target")
                    source_embedding = target_face.get('embedding')
                
                # Run inference
                swapped_patch = self._forward_swap(patch_prepared, source_embedding, model_type)
                swapped_patch = self._normalize_crop_frame(swapped_patch)
                swapped_patches.append(swapped_patch)
            
            # Merge patches back
            if pixel_boost_total > 1:
                crop_frame_swapped = explode_pixel_boost(swapped_patches, pixel_boost_total, model_size, pixel_boost_size)
            else:
                crop_frame_swapped = swapped_patches[0]
            
            # Create box mask for blending with blur
            mask = self._create_box_mask(crop_frame_swapped.shape[:2], face_mask_blur)
            
            # Combine with occluder mask if created
            if occluder_mask is not None:
                # Resize to match mask size if needed
                if occluder_mask.shape != mask.shape:
                    occluder_mask = cv2.resize(occluder_mask, (mask.shape[1], mask.shape[0]))
                # Occluder mask: 1 = visible (face), 0 = occluded
                # Multiply masks to keep only visible regions
                # print(f"[LocalFaceSwapper] Applying occluder mask: before mean={mask.mean():.3f}")
                mask = mask * occluder_mask
                # print(f"[LocalFaceSwapper] After occluder: mean={mask.mean():.3f}")
            
            # Combine with parser mask if created
            if parser_mask is not None:
                # Resize to match mask size if needed
                if parser_mask.shape != mask.shape:
                    parser_mask = cv2.resize(parser_mask, (mask.shape[1], mask.shape[0]))
                # Parser mask: 1 = face region, 0 = background/hair
                # Multiply masks to keep only face regions
                # print(f"[LocalFaceSwapper] Applying parser mask")
                mask = mask * parser_mask
                # print(f"[LocalFaceSwapper] After parser: mean={mask.mean():.3f}")
            
            # Paste back into original image
            result = self._paste_back(target_image, crop_frame_swapped, mask, affine_matrix)
            
            return result
            
        except Exception as e:
            print(f"[LocalFaceSwapper] Error swapping face: {e}")
            import traceback
            traceback.print_exc()
            return target_image
    
    def _warp_face(
        self,
        image: VisionFrame,
        landmarks: NDArray,
        template_name: str,
        size: Tuple[int, int]
    ) -> Tuple[VisionFrame, NDArray]:
        """Warp face using landmarks to standard template."""
        template = WARP_TEMPLATES.get(template_name)
        if template is None:
            template = WARP_TEMPLATES['arcface_112_v2']
        
        # Scale template to target size
        template_scaled = template * np.array([size[0], size[1]])
        
        # Estimate affine transform
        affine_matrix = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32),
            template_scaled,
            method=cv2.LMEDS
        )[0]
        
        # Warp image
        warped = cv2.warpAffine(
            image,
            affine_matrix,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return warped, affine_matrix
    
    def _prepare_crop_frame(self, frame: VisionFrame) -> NDArray:
        """Prepare crop frame for model input."""
        # Convert BGR to RGB and normalize to [-1, 1]
        prepared = frame[:, :, ::-1].astype(np.float32) / 255.0
        prepared = (prepared - 0.5) / 0.5  # Scale to [-1, 1]
        
        # Transpose to CHW format and add batch dimension
        prepared = prepared.transpose(2, 0, 1)
        prepared = np.expand_dims(prepared, 0)
        
        return prepared
    
    def _normalize_crop_frame(self, frame: NDArray) -> VisionFrame:
        """Normalize model output back to image format."""
        # Remove batch dimension and transpose to HWC
        frame = frame.transpose(1, 2, 0)
        
        # Denormalize from [-1, 1] to [0, 255]
        frame = (frame * 0.5 + 0.5) * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR
        frame = frame[:, :, ::-1]
        
        return frame
    
    def _forward_swap(
        self,
        target_frame: NDArray,
        source_embedding: Optional[NDArray],
        model_type: str
    ) -> NDArray:
        """Run forward pass through the swapper model."""
        inputs = {}
        
        # Prepare inputs based on model type
        for model_input in self.model_session.get_inputs():
            if model_input.name == 'target':
                inputs['target'] = target_frame
            elif model_input.name == 'source':
                if source_embedding is not None:
                    # Reshape embedding for model
                    if len(source_embedding.shape) == 1:
                        source_embedding = np.expand_dims(source_embedding, 0)
                    inputs['source'] = source_embedding.astype(np.float32)
                else:
                    # Use zero embedding as fallback
                    inputs['source'] = np.zeros((1, 512), dtype=np.float32)
        
        # Run inference
        outputs = self.model_session.run(None, inputs)
        result = outputs[0][0]  # Get first output, remove batch dimension
        
        return result
    
    def _create_box_mask(self, size: Tuple[int, int], face_mask_blur: float = 0.3, padding: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> NDArray:
        """Create a box mask for blending with blur (matching facefusion implementation)."""
        height, width = size
        
        # Calculate blur amount based on mask size (following facefusion formula)
        blur_amount = int(width * 0.5 * face_mask_blur)
        blur_area = max(blur_amount // 2, 1)
        
        # Create base mask
        mask = np.ones((height, width), dtype=np.float32)
        
        # Apply padding to mask edges (create fade zones)
        if blur_area > 0:
            # Top
            mask[:blur_area, :] = 0
            # Bottom
            mask[-blur_area:, :] = 0
            # Left
            mask[:, :blur_area] = 0
            # Right
            mask[:, -blur_area:] = 0
        
        # Apply Gaussian blur for smooth blending
        if blur_amount > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
        
        return mask
    
    def _paste_back(
        self,
        target_image: VisionFrame,
        crop_frame: VisionFrame,
        mask: NDArray,
        affine_matrix: NDArray
    ) -> VisionFrame:
        """Paste swapped crop back into original image with blending."""
        # Calculate inverse transform
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        
        # Get paste bounding box
        crop_height, crop_width = crop_frame.shape[:2]
        target_height, target_width = target_image.shape[:2]
        
        # Transform corners of crop to find paste region
        corners = np.array([
            [0, 0],
            [crop_width, 0],
            [crop_width, crop_height],
            [0, crop_height]
        ], dtype=np.float32)
        
        corners_transformed = cv2.transform(corners.reshape(1, -1, 2), inverse_matrix).reshape(-1, 2)
        
        # Get bounding box
        x_min = int(np.floor(corners_transformed[:, 0].min()))
        y_min = int(np.floor(corners_transformed[:, 1].min()))
        x_max = int(np.ceil(corners_transformed[:, 0].max()))
        y_max = int(np.ceil(corners_transformed[:, 1].max()))
        
        # Clip to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(target_width, x_max)
        y_max = min(target_height, y_max)
        
        paste_width = x_max - x_min
        paste_height = y_max - y_min
        
        if paste_width <= 0 or paste_height <= 0:
            return target_image
        
        # Warp crop and mask back to original space
        paste_matrix = inverse_matrix.copy()
        paste_matrix[0, 2] -= x_min
        paste_matrix[1, 2] -= y_min
        
        warped_crop = cv2.warpAffine(
            crop_frame,
            paste_matrix,
            (paste_width, paste_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        warped_mask = cv2.warpAffine(
            mask,
            paste_matrix,
            (paste_width, paste_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        # Ensure mask is 3 channels
        if len(warped_mask.shape) == 2:
            warped_mask = np.expand_dims(warped_mask, -1)
        
        # Blend
        result = target_image.copy()
        paste_region = result[y_min:y_max, x_min:x_max]
        blended = paste_region * (1 - warped_mask) + warped_crop * warped_mask
        result[y_min:y_max, x_min:x_max] = blended.astype(np.uint8)
        
        return result


# Global instance
_swapper_instance = None


def get_local_swapper(model_name: str = 'hyperswap_1c_256') -> LocalFaceSwapper:
    """Get or create local face swapper instance."""
    global _swapper_instance
    if _swapper_instance is None or _swapper_instance.model_name != model_name:
        _swapper_instance = LocalFaceSwapper(model_name)
    return _swapper_instance

