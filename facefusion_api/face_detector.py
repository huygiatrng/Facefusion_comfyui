"""
Local face detection using ONNX models - matches facefusion implementation.
"""
import os
from typing import List, Optional, Tuple
from functools import lru_cache

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from .utils import (
    VisionFrame, BoundingBox, FaceLandmark5, Face,
    create_face_dict, sort_faces_by_order, select_face_by_position,
    find_matching_faces, get_model_path, ensure_model_exists
)


class FaceDetector:
    """Face detector using ONNX models - simplified to match facefusion."""
    
    MODEL_URLS = {
        'scrfd_2.5g': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/scrfd_2.5g.onnx',
        'arcface_w600k_r50': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.onnx',
    }
    
    def __init__(self, detector_model: str = 'scrfd_2.5g', recognition_model: str = 'arcface_w600k_r50'):
        self.detector_model_name = detector_model
        self.recognition_model_name = recognition_model
        self.detector_session = None
        self.recognition_session = None
        self.detector_size = (640, 640)
        
    def initialize(self) -> bool:
        """Initialize the detector and recognition models."""
        try:
            detector_path = get_model_path(f'{self.detector_model_name}.onnx')
            recognition_path = get_model_path(f'{self.recognition_model_name}.onnx')
            
            if not os.path.exists(detector_path):
                print(f"Downloading face detector model: {self.detector_model_name}")
                if not ensure_model_exists(
                    f'{self.detector_model_name}.onnx',
                    self.MODEL_URLS.get(self.detector_model_name)
                ):
                    return False
            
            if not os.path.exists(recognition_path):
                print(f"Downloading face recognition model: {self.recognition_model_name}")
                if not ensure_model_exists(
                    f'{self.recognition_model_name}.onnx',
                    self.MODEL_URLS.get(self.recognition_model_name)
                ):
                    return False
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.detector_session = ort.InferenceSession(detector_path, providers=providers)
            self.recognition_session = ort.InferenceSession(recognition_path, providers=providers)
            
            print(f"Face detector initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing face detector: {e}")
            return False
    
    def detect_faces(self, image: VisionFrame, score_threshold: float = 0.3) -> List[Face]:
        """Detect faces - simplified to match facefusion implementation."""
        if self.detector_session is None:
            if not self.initialize():
                return []
        
        try:
            # Debug logging (commented out for cleaner output)
            # print(f"[FaceDetector] Input image shape: {image.shape}, dtype: {image.dtype}")
            # print(f"[FaceDetector] Score threshold: {score_threshold}")
            
            # Prepare input - matching facefusion's approach
            detect_frame = self._prepare_detect_frame(image)
            # print(f"[FaceDetector] Prepared frame shape: {detect_frame.shape}")
            
            detect_frame = self._normalize_detect_frame(detect_frame)
            # print(f"[FaceDetector] Normalized frame shape: {detect_frame.shape}, range: [{detect_frame.min():.3f}, {detect_frame.max():.3f}]")
            
            # Run detection
            detection = self.detector_session.run(None, {'input': detect_frame})
            # print(f"[FaceDetector] Detection outputs: {len(detection)} arrays, shapes: {[d.shape for d in detection]}")
            
            # Parse results - matching facefusion's approach
            bboxes, scores, landmarks = self._parse_scrfd_detection(
                detection, image, score_threshold
            )
            
            # print(f"[FaceDetector] Found {len(bboxes)} faces with scores: {scores}")
            
            # Create face dictionaries
            faces = []
            for bbox, score, landmark in zip(bboxes, scores, landmarks):
                face = create_face_dict(bbox, landmark, score)
                # Get embedding
                embedding = self._get_face_embedding(image, face)
                if embedding is not None:
                    face['embedding'] = embedding
                faces.append(face)
            
            return faces
        except Exception as e:
            print(f"Error detecting faces: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _prepare_detect_frame(self, image: VisionFrame) -> VisionFrame:
        """Prepare image for detection - resize and maintain aspect ratio."""
        height, width = image.shape[:2]
        detect_width, detect_height = self.detector_size
        
        # Calculate scale to fit within detector_size
        scale = min(detect_height / height, detect_width / width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Resize
        if new_height != height or new_width != width:
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized = image.copy()
        
        # Create padded frame
        padded = np.zeros((detect_height, detect_width, 3), dtype=np.uint8)
        padded[:new_height, :new_width] = resized
        
        return padded
    
    def _normalize_detect_frame(self, frame: VisionFrame) -> NDArray:
        """Normalize frame for detection - match facefusion's [-1, 1] range."""
        normalized = frame.astype(np.float32)
        # Normalize to [-1, 1] range (facefusion uses 128.0 NOT 127.5!)
        normalized = (normalized - 127.5) / 128.0
        # Transpose to CHW format
        normalized = np.transpose(normalized, (2, 0, 1))
        normalized = np.expand_dims(normalized, 0)
        return normalized
    
    def _parse_scrfd_detection(
        self,
        detection: List[NDArray],
        original_image: VisionFrame,
        score_threshold: float
    ) -> Tuple[List[BoundingBox], List[float], List[FaceLandmark5]]:
        """Parse SCRFD detection outputs - match facefusion's approach."""
        bboxes_list = []
        scores_list = []
        landmarks_list = []
        
        feature_strides = [8, 16, 32]
        feature_map_channel = 3
        anchor_total = 2
        
        orig_height, orig_width = original_image.shape[:2]
        detect_width, detect_height = self.detector_size
        
        # Calculate actual scale used
        scale = min(detect_height / orig_height, detect_width / orig_width)
        temp_height = int(orig_height * scale)
        temp_width = int(orig_width * scale)
        
        ratio_height = orig_height / temp_height
        ratio_width = orig_width / temp_width
        
        for idx, feature_stride in enumerate(feature_strides):
            scores_raw = detection[idx]
            bbox_raw = detection[idx + feature_map_channel]
            landmark_raw = detection[idx + feature_map_channel * 2]
            
            # Debug: show max score for each stride (commented out for cleaner output)
            # max_score = scores_raw.max() if scores_raw.size > 0 else 0
            # print(f"[FaceDetector] Stride {feature_stride}: max_score={max_score:.4f}, threshold={score_threshold:.4f}")
            
            # Filter by score threshold
            keep_indices = np.where(scores_raw >= score_threshold)[0]
            
            if len(keep_indices) > 0:
                # print(f"[FaceDetector] Stride {feature_stride}: {len(keep_indices)} faces above threshold")
                stride_height = detect_height // feature_stride
                stride_width = detect_width // feature_stride
                
                # Create anchors - match facefusion's approach
                anchors = self._create_anchors(feature_stride, anchor_total, stride_height, stride_width)
                
                # Scale predictions
                bbox_preds = bbox_raw[keep_indices] * feature_stride
                landmark_preds = landmark_raw[keep_indices] * feature_stride
                
                # Decode bounding boxes
                anchor_points = anchors[keep_indices]
                x1 = anchor_points[:, 0] - bbox_preds[:, 0]
                y1 = anchor_points[:, 1] - bbox_preds[:, 1]
                x2 = anchor_points[:, 0] + bbox_preds[:, 2]
                y2 = anchor_points[:, 1] + bbox_preds[:, 3]
                
                # Apply ratio to get original image coordinates
                bboxes = np.stack([x1 * ratio_width, y1 * ratio_height, 
                                   x2 * ratio_width, y2 * ratio_height], axis=1)
                
                # Decode landmarks
                landmarks = []
                for i in range(5):
                    x = anchor_points[:, 0] + landmark_preds[:, i * 2]
                    y = anchor_points[:, 1] + landmark_preds[:, i * 2 + 1]
                    landmarks.append(x * ratio_width)
                    landmarks.append(y * ratio_height)
                landmarks = np.stack(landmarks, axis=1).reshape(-1, 5, 2)
                
                # Get scores
                scores = scores_raw[keep_indices].flatten()
                
                bboxes_list.append(bboxes)
                scores_list.append(scores)
                landmarks_list.extend(landmarks)
        
        # Concatenate all levels
        if len(bboxes_list) > 0:
            all_bboxes = np.vstack(bboxes_list)
            all_scores = np.concatenate(scores_list)
            all_landmarks = landmarks_list
            
            # Apply NMS
            keep = self._nms(all_bboxes, all_scores, 0.4)
            
            return (
                [all_bboxes[i] for i in keep],
                [float(all_scores[i]) for i in keep],
                [all_landmarks[i] for i in keep]
            )
        
        return [], [], []
    
    @lru_cache(maxsize=100)
    def _create_anchors(self, feature_stride: int, anchor_total: int, 
                        stride_height: int, stride_width: int) -> NDArray:
        """Create anchor points - match facefusion's approach."""
        x, y = np.mgrid[:stride_width, :stride_height]
        anchors = np.stack((y, x), axis=-1)
        anchors = (anchors * feature_stride).reshape((-1, 2))
        anchors = np.stack([anchors] * anchor_total, axis=1).reshape((-1, 2))
        return anchors.astype(np.float32)
    
    def _nms(self, boxes: NDArray, scores: NDArray, threshold: float) -> List[int]:
        """Non-maximum suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _get_face_embedding(self, image: VisionFrame, face: Face) -> Optional[NDArray]:
        """Extract face embedding for recognition."""
        if self.recognition_session is None:
            return None
        
        try:
            # Align face using landmarks
            aligned_face = self._align_face(image, face['landmarks'])
            if aligned_face is None:
                return None
            
            # Prepare input
            input_face = cv2.resize(aligned_face, (112, 112))
            input_face = input_face.astype(np.float32)
            input_face = (input_face - 127.5) / 127.5
            input_face = np.transpose(input_face, (2, 0, 1))
            input_face = np.expand_dims(input_face, 0)
            
            # Run recognition
            outputs = self.recognition_session.run(None, {'input': input_face})
            
            embedding = outputs[0].flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def _align_face(self, image: VisionFrame, landmarks: FaceLandmark5) -> Optional[VisionFrame]:
        """Align face using landmarks."""
        try:
            # ArcFace template
            reference_landmarks = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]
            ], dtype=np.float32)
            
            matrix = cv2.estimateAffinePartial2D(
                landmarks.astype(np.float32),
                reference_landmarks,
                method=cv2.LMEDS
            )[0]
            
            if matrix is None:
                return None
            
            aligned = cv2.warpAffine(image, matrix, (112, 112),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
            
            return aligned
        except:
            return None


# Global detector instance
_detector_instance = None


def get_face_detector() -> FaceDetector:
    """Get global face detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector()
    return _detector_instance


def detect_faces(
    image: VisionFrame,
    score_threshold: float = 0.3,
    sort_order: str = 'large-small'
) -> List[Face]:
    """Detect and sort faces in an image."""
    detector = get_face_detector()
    faces = detector.detect_faces(image, score_threshold)
    return sort_faces_by_order(faces, sort_order)


def select_faces(
    target_image: VisionFrame,
    mode: str = 'one',
    position: int = 0,
    reference_face: Optional[Face] = None,
    distance_threshold: float = 0.6,
    score_threshold: float = 0.3
) -> List[Face]:
    """Select faces from target image based on mode."""
    target_faces = detect_faces(target_image, score_threshold)
    
    if mode == 'many':
        return target_faces
    
    if mode == 'one':
        selected = select_face_by_position(target_faces, position)
        return [selected] if selected else []
    
    if mode == 'reference' and reference_face:
        return find_matching_faces(reference_face, target_faces, distance_threshold)
    
    return []
