"""
Local face swapper using ONNX models.
"""
import os
from typing import Optional, Tuple, List, TYPE_CHECKING

import cv2
import numpy as np
import onnxruntime as ort
import onnx
from onnx import numpy_helper
from numpy.typing import NDArray

from ..utils import VisionFrame, Face, get_model_path, ensure_model_exists, implode_pixel_boost, explode_pixel_boost
from .constants import MODEL_URLS, MODEL_CONFIGS, WARP_TEMPLATES, FACE_MASK_AREA_SET, FACE_MASK_REGION_SET

if TYPE_CHECKING:
    from .occluder import FaceOccluder
    from .parser import FaceParser


class LocalFaceSwapper:
    """Local face swapping using ONNX models."""
    
    def __init__(self, model_name: str = 'hyperswap_1c_256'):
        self.model_name = model_name
        self.model_session = None
        self.embedding_converter_session = None
        self.model_initializer = None
        self.model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['hyperswap_1c_256'])
        self.model_initializer = None
        
    def initialize(self) -> bool:
        """Initialize the face swapper model and embedding converter if needed."""
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
            
            # Load model_initializer for inswapper models
            model_type = self.model_config.get('type')
            if model_type == 'inswapper':
                import onnx
                model = onnx.load(model_path)
                self.model_initializer = onnx.numpy_helper.to_array(model.graph.initializer[-1])
                print(f"[LocalFaceSwapper] Loaded model_initializer for inswapper: shape={self.model_initializer.shape}")
            
            # Load embedding converter for models that need it
            converter_name = self.model_config.get('converter')
            if converter_name:
                converter_path = get_model_path(f'{converter_name}.onnx')
                if not os.path.exists(converter_path):
                    print(f"[LocalFaceSwapper] Downloading embedding converter: {converter_name}")
                    converter_url = MODEL_URLS.get(converter_name)
                    if converter_url and ensure_model_exists(f'{converter_name}.onnx', converter_url):
                        self.embedding_converter_session = ort.InferenceSession(converter_path, providers=providers)
                        print(f"[LocalFaceSwapper] Loaded embedding converter: {converter_name}")
                else:
                    self.embedding_converter_session = ort.InferenceSession(converter_path, providers=providers)
                    print(f"[LocalFaceSwapper] Loaded embedding converter: {converter_name}")
            
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
        face_parser: Optional['FaceParser'] = None,
        source_image: Optional[VisionFrame] = None,
        face_mask_types: Optional[List[str]] = None,
        face_mask_areas: Optional[List[str]] = None,
        face_mask_regions: Optional[List[str]] = None,
        face_mask_padding: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ) -> VisionFrame:
        """Swap a single face in the target image with multi-mask support."""
        if self.model_session is None:
            if not self.initialize():
                return target_image
        
        # Default mask types
        if face_mask_types is None:
            face_mask_types = ['box']
        
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
            
            # Prepare source frame for models that need it (blendswap, uniface)
            if model_type in ['blendswap', 'uniface'] and source_image is not None:
                source_face['source_frame'] = self._prepare_source_frame(source_image, source_face, model_type)
            
            # Warp target face to pixel boost size
            crop_frame_original, affine_matrix = self._warp_face(
                target_image,
                target_face['landmarks'],
                model_template,
                pixel_boost_size
            )
            
            # Collect all crop masks based on face_mask_types (before swapping, just like facefusion-master)
            crop_masks = []
            
            # BOX mask
            if 'box' in face_mask_types:
                box_mask = self._create_box_mask(crop_frame_original.shape[:2], face_mask_blur, face_mask_padding)
                crop_masks.append(box_mask)
            
            # OCCLUSION mask (using face occluder model)
            if 'occlusion' in face_mask_types and face_occluder is not None:
                occluder_mask = face_occluder.create_occlusion_mask(crop_frame_original)
                if occluder_mask is not None:
                    crop_masks.append(occluder_mask)
            
            # Split into patches for pixel boost
            if pixel_boost_total > 1:
                crop_patches = implode_pixel_boost(crop_frame_original, pixel_boost_total, model_size)
            else:
                crop_patches = [crop_frame_original]
            
            # Process each patch through the model
            swapped_patches = []
            for patch in crop_patches:
                patch_prepared = self._prepare_crop_frame(patch)
                source_embedding = self._prepare_source_embedding(source_face, target_face, model_type)
                swapped_patch = self._forward_swap(patch_prepared, source_embedding, model_type, source_face)
                swapped_patch = self._normalize_crop_frame(swapped_patch)
                swapped_patches.append(swapped_patch)
            
            # Merge patches back
            if pixel_boost_total > 1:
                crop_frame_swapped = explode_pixel_boost(swapped_patches, pixel_boost_total, model_size, pixel_boost_size)
            else:
                crop_frame_swapped = swapped_patches[0]
            
            # AREA mask (using face landmarks - created after swapping per facefusion-master logic)
            if 'area' in face_mask_types and face_mask_areas:
                # Transform target face landmarks to crop space
                face_landmark_68 = target_face.get('landmarks_68')
                if face_landmark_68 is not None:
                    face_landmark_68_transformed = cv2.transform(
                        face_landmark_68.reshape(1, -1, 2).astype(np.float32), 
                        affine_matrix
                    ).reshape(-1, 2)
                    area_mask = self._create_area_mask(crop_frame_swapped, face_landmark_68_transformed, face_mask_areas)
                    crop_masks.append(area_mask)
            
            # REGION mask (using face parser model - created after swapping per facefusion-master logic)
            if 'region' in face_mask_types and face_parser is not None and face_mask_regions:
                region_mask = face_parser.create_region_mask(crop_frame_swapped, face_mask_regions)
                if region_mask is not None:
                    crop_masks.append(region_mask)
            
            # Combine all masks using minimum (intersection)
            if crop_masks:
                combined_mask = np.minimum.reduce(crop_masks).clip(0, 1)
            else:
                # Fallback to simple box mask if no masks specified
                combined_mask = self._create_box_mask(crop_frame_swapped.shape[:2], face_mask_blur, face_mask_padding)
            
            # Paste back into original image
            result = self._paste_back(target_image, crop_frame_swapped, combined_mask, affine_matrix)
            
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
        """Prepare crop frame for model input with model-specific normalization (matches facefusion)."""
        # Convert BGR to RGB 
        prepared = frame[:, :, ::-1].astype(np.float32) / 255.0
        
        # Get model-specific normalization parameters
        mean = np.array(self.model_config.get('mean', [0.5, 0.5, 0.5]), dtype=np.float32)
        std = np.array(self.model_config.get('std', [0.5, 0.5, 0.5]), dtype=np.float32)
        
        # Apply normalization: (x - mean) / std
        prepared = (prepared - mean) / std
        
        # Transpose to CHW format and add batch dimension
        prepared = prepared.transpose(2, 0, 1)
        prepared = np.expand_dims(prepared, 0)
        
        return prepared
    
    def _prepare_source_embedding(self, source_face: Face, target_face: Face, model_type: str) -> Optional[NDArray]:
        """Prepare source embedding based on model type - matches facefusion implementation."""
        embedding = source_face.get('embedding')  # Raw embedding (NOT normalized)
        embedding_norm = source_face.get('embedding_norm')  # Normalized embedding
        
        if embedding is None:
            print(f"[LocalFaceSwapper] Warning: No source embedding")
            return None
        
        # Different models need different embedding formats
        if model_type == 'hyperswap':
            # HyperSwap uses normalized embedding directly (embedding_norm)
            if embedding_norm is not None:
                source_embedding = embedding_norm.reshape((1, -1))
            else:
                source_embedding = (embedding / np.linalg.norm(embedding)).reshape((1, -1))
        
        elif model_type == 'inswapper':
            # InSwapper uses model_initializer transformation on RAW embedding
            if self.model_initializer is not None:
                source_embedding = embedding.reshape((1, -1))
                source_embedding = np.dot(source_embedding, self.model_initializer) / np.linalg.norm(source_embedding)
            else:
                print("[LocalFaceSwapper] Warning: No model_initializer for inswapper")
                source_embedding = embedding.reshape((1, -1))
        
        elif model_type == 'ghost':
            # Ghost uses RAW embedding, converts it, returns converted (NOT normalized)
            if self.embedding_converter_session is not None:
                source_embedding_reshaped = embedding.reshape(-1, 512).astype(np.float32)
                outputs = self.embedding_converter_session.run(None, {'input': source_embedding_reshaped})
                converted_embedding = outputs[0].ravel()
                # Ghost uses converted embedding WITHOUT normalization
                source_embedding = converted_embedding.reshape((1, -1))
            else:
                print(f"[LocalFaceSwapper] Warning: No embedding converter for ghost")
                source_embedding = embedding.reshape((1, -1))
        
        elif model_type in ['hififace', 'simswap']:
            # SimSwap/HifiFace use RAW embedding, convert it, return normalized
            if self.embedding_converter_session is not None:
                source_embedding_reshaped = embedding.reshape(-1, 512).astype(np.float32)
                outputs = self.embedding_converter_session.run(None, {'input': source_embedding_reshaped})
                converted_embedding = outputs[0].ravel()
                # SimSwap/HifiFace use converted embedding WITH normalization
                source_embedding_norm = converted_embedding / np.linalg.norm(converted_embedding)
                source_embedding = source_embedding_norm.reshape((1, -1))
            else:
                print(f"[LocalFaceSwapper] Warning: No embedding converter for {model_type}")
                source_embedding = (embedding / np.linalg.norm(embedding)).reshape((1, -1))
        
        else:
            # Default: use normalized embedding (blendswap, uniface, etc.)
            if embedding_norm is not None:
                source_embedding = embedding_norm.reshape((1, -1))
            else:
                source_embedding = (embedding / np.linalg.norm(embedding)).reshape((1, -1))
        
        return source_embedding.astype(np.float32)
    
    def _prepare_source_frame(self, source_image: VisionFrame, source_face: Face, model_type: str) -> NDArray:
        """Prepare source frame for blendswap/uniface models."""
        # Get warp template and size based on model type
        if model_type == 'blendswap':
            template_name = 'arcface_112_v2'
            size = (112, 112)
        elif model_type == 'uniface':
            template_name = 'ffhq_512'
            size = (256, 256)
        else:
            return None
        
        # Warp source face
        warped_source, _ = self._warp_face(
            source_image,
            source_face['landmarks'],
            template_name,
            size
        )
        
        # Convert to model input format (same as prepare_crop_frame but for source)
        prepared = warped_source[:, :, ::-1].astype(np.float32) / 255.0
        prepared = prepared.transpose(2, 0, 1)
        prepared = np.expand_dims(prepared, 0)
        
        return prepared

    def _get_model_initializer(self) -> Optional[NDArray]:
        """Load initializer matrix from ONNX model (needed for inswapper)."""
        if self.model_initializer is not None:
            return self.model_initializer
        
        try:
            model_path = get_model_path(f'{self.model_name}.onnx')
            if not os.path.exists(model_path):
                return None
            model = onnx.load(model_path)
            if not model.graph.initializer:
                return None
            # Use last initializer (matches facefusion implementation)
            self.model_initializer = numpy_helper.to_array(model.graph.initializer[-1])
        except Exception as e:
            print(f"[LocalFaceSwapper] Warning: Unable to load model initializer: {e}")
            self.model_initializer = None
        
        return self.model_initializer
    
    def _normalize_crop_frame(self, frame: NDArray) -> VisionFrame:
        """Normalize model output back to image format with model-specific denormalization (matches facefusion)."""
        # Remove batch dimension and transpose to HWC
        frame = frame.transpose(1, 2, 0)
        
        model_type = self.model_config.get('type', 'hyperswap')
        
        # Only certain models need reverse normalization
        # simswap, inswapper, blendswap do NOT get reverse normalization
        if model_type in ['ghost', 'hififace', 'hyperswap', 'uniface']:
            # Apply reverse normalization: x * std + mean
            mean = np.array(self.model_config.get('mean', [0.5, 0.5, 0.5]), dtype=np.float32)
            std = np.array(self.model_config.get('std', [0.5, 0.5, 0.5]), dtype=np.float32)
            frame = frame * std + mean
        
        # Clip to [0, 1] and scale to [0, 255]
        frame = np.clip(frame, 0, 1)
        frame = frame * 255.0
        frame = frame.astype(np.uint8)
        
        # Convert RGB to BGR
        frame = frame[:, :, ::-1]
        
        return frame
    
    def _forward_swap(
        self,
        target_frame: NDArray,
        source_embedding: Optional[NDArray],
        model_type: str,
        source_face: Optional[Face] = None
    ) -> NDArray:
        """Run forward pass through the swapper model."""
        inputs = {}
        
        # Prepare inputs based on model type
        for model_input in self.model_session.get_inputs():
            if model_input.name == 'target':
                inputs['target'] = target_frame
            elif model_input.name == 'source':
                # Different models need different source formats
                if model_type in ['blendswap', 'uniface']:
                    # These models need a warped source face image, not embedding
                    if source_face is not None and 'source_frame' in source_face:
                        # Use pre-prepared source frame
                        inputs['source'] = source_face['source_frame']
                    else:
                        # Fallback: use zero frame
                        print(f"[LocalFaceSwapper] Warning: {model_type} needs source_frame, using fallback")
                        inputs['source'] = np.zeros((1, 3, 112, 112), dtype=np.float32)
                else:
                    # Other models use embeddings
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
        """Create a box mask for blending with blur and padding (matching facefusion implementation).
        
        Args:
            size: (height, width) of the mask
            face_mask_blur: Blur amount (0.0-1.0)
            padding: (top, right, bottom, left) padding percentages
        """
        height, width = size
        
        # Calculate blur amount based on mask size (following facefusion formula)
        blur_amount = int(width * 0.5 * face_mask_blur)
        blur_area = max(blur_amount // 2, 1)
        
        # Create base mask
        mask = np.ones((height, width), dtype=np.float32)
        
        # Apply padding to mask edges (create fade zones) - matching facefusion
        # Top padding
        top_pad = max(blur_area, int(height * padding[0] / 100))
        mask[:top_pad, :] = 0
        # Bottom padding  
        bottom_pad = max(blur_area, int(height * padding[2] / 100))
        mask[-bottom_pad:, :] = 0
        # Left padding
        left_pad = max(blur_area, int(width * padding[3] / 100))
        mask[:, :left_pad] = 0
        # Right padding
        right_pad = max(blur_area, int(width * padding[1] / 100))
        mask[:, -right_pad:] = 0
        
        # Apply Gaussian blur for smooth blending
        if blur_amount > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
        
        return mask
    
    def _create_area_mask(self, crop_vision_frame: VisionFrame, face_landmark_68: NDArray, face_mask_areas: List[str]) -> NDArray:
        """Create area mask based on face landmarks (matching facefusion implementation).
        
        Args:
            crop_vision_frame: The cropped face frame
            face_landmark_68: 68-point facial landmarks transformed to crop space
            face_mask_areas: List of areas to include ['upper-face', 'lower-face', 'mouth']
        """
        crop_size = crop_vision_frame.shape[:2][::-1]  # (width, height)
        landmark_points = []
        
        # Collect landmark indices for each requested area
        for face_mask_area in face_mask_areas:
            if face_mask_area in FACE_MASK_AREA_SET:
                landmark_points.extend(FACE_MASK_AREA_SET.get(face_mask_area))
        
        if not landmark_points or face_landmark_68 is None:
            # Return full mask if no valid areas or landmarks
            return np.ones(crop_size[::-1], dtype=np.float32)
        
        # Get unique landmark points
        landmark_points = list(set(landmark_points))
        
        # Create convex hull from landmark points
        try:
            convex_hull = cv2.convexHull(face_landmark_68[landmark_points].astype(np.int32))
            area_mask = np.zeros(crop_size[::-1], dtype=np.float32)
            cv2.fillConvexPoly(area_mask, convex_hull, 1.0)
            
            # Apply Gaussian blur for smooth edges (matching facefusion)
            area_mask = (cv2.GaussianBlur(area_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
            return area_mask
        except Exception as e:
            print(f"[LocalFaceSwapper] Warning: Could not create area mask: {e}")
            return np.ones(crop_size[::-1], dtype=np.float32)
    
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

