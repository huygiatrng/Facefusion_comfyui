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

from ..utils import (
    VisionFrame, BoundingBox, FaceLandmark5, Face,
    create_face_dict, sort_faces_by_order, select_face_by_position,
    find_matching_faces, get_model_path, ensure_model_exists
)


class FaceDetector:
    """Face detector using ONNX models - simplified to match facefusion."""
    
    MODEL_URLS = {
        'scrfd_2.5g': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/scrfd_2.5g.onnx',
        'retinaface_10g': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface_10g.onnx',
        'yoloface_8n': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx',
        'yunet_2023_mar': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/yunet_2023_mar.onnx',
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
        """Detect faces - supports multiple detector models."""
        if self.detector_session is None:
            if not self.initialize():
                return []
        
        try:
            # Prepare input - matching facefusion's approach
            detect_frame = self._prepare_detect_frame(image)
            
            # Normalize based on model type
            if self.detector_model_name in ['yoloface_8n']:
                detect_frame = self._normalize_detect_frame(detect_frame, [0, 1])
            elif self.detector_model_name in ['yunet_2023_mar']:
                detect_frame = self._normalize_detect_frame(detect_frame, [0, 255])
            else:  # scrfd, retinaface
                detect_frame = self._normalize_detect_frame(detect_frame, [-1, 1])
            
            # Run detection
            detection = self.detector_session.run(None, {'input': detect_frame})
            
            # Parse results based on model type
            if self.detector_model_name == 'yoloface_8n':
                bboxes, scores, landmarks = self._parse_yolo_face_detection(
                    detection, image, score_threshold
                )
            elif self.detector_model_name == 'yunet_2023_mar':
                bboxes, scores, landmarks = self._parse_yunet_detection(
                    detection, image, score_threshold
                )
            else:  # scrfd, retinaface (same format)
                bboxes, scores, landmarks = self._parse_scrfd_detection(
                    detection, image, score_threshold
                )
            
            # Create face dictionaries
            faces = []
            for bbox, score, landmark in zip(bboxes, scores, landmarks):
                face = create_face_dict(bbox, landmark, score)
                # Get embedding (raw + normalized)
                embeddings = self._get_face_embedding(image, face)
                if embeddings is not None:
                    embedding, embedding_norm = embeddings
                    face['embedding'] = embedding
                    face['embedding_norm'] = embedding_norm
                faces.append(face)
            
            return faces
        except Exception as e:
            print(f"Error detecting faces: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _prepare_detect_frame(self, image: VisionFrame) -> VisionFrame:
        """Prepare image for detection - resize, pad, and convert to CHW format."""
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
        
        # Create padded frame (HWC format)
        padded = np.zeros((detect_height, detect_width, 3), dtype=np.float32)
        padded[:new_height, :new_width] = resized
        
        # Convert to CHW format and add batch dimension: (1, C, H, W)
        detect_frame = np.transpose(padded, (2, 0, 1))
        detect_frame = np.expand_dims(detect_frame, 0).astype(np.float32)
        
        return detect_frame
    
    def _normalize_detect_frame(self, frame: VisionFrame, normalize_range: list = [-1, 1]) -> NDArray:
        """Normalize frame for detection - supports different ranges."""
        # Frame is already in CHW format from _prepare_detect_frame
        if normalize_range == [-1, 1]:
            # SCRFD, RetinaFace: [-1, 1] range
            return (frame - 127.5) / 128.0
        elif normalize_range == [0, 1]:
            # YOLO Face: [0, 1] range
            return frame / 255.0
        elif normalize_range == [0, 255]:
            # YuNet: [0, 255] range (no normalization)
            return frame
        return frame
    
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
    
    def _parse_yolo_face_detection(
        self,
        detection: List[NDArray],
        original_image: VisionFrame,
        score_threshold: float
    ) -> Tuple[List[BoundingBox], List[float], List[FaceLandmark5]]:
        """Parse YOLO Face detection outputs - match facefusion's approach."""
        bboxes_list = []
        scores_list = []
        landmarks_list = []
        
        orig_height, orig_width = original_image.shape[:2]
        detect_width, detect_height = self.detector_size
        
        # Calculate actual scale used
        scale = min(detect_height / orig_height, detect_width / orig_width)
        temp_height = int(orig_height * scale)
        temp_width = int(orig_width * scale)
        
        ratio_height = orig_height / temp_height
        ratio_width = orig_width / temp_width
        
        # YOLO Face output format: (1, 16800, 15) -> (bboxes[4], score[1], landmarks[10])
        detection_array = np.squeeze(detection[0]).T  # Shape: (N, 15)
        bboxes_raw = detection_array[:, :4]
        scores_raw = detection_array[:, 4]
        landmarks_raw = detection_array[:, 5:]
        
        # Filter by score threshold
        keep_indices = np.where(scores_raw > score_threshold)[0]
        
        if len(keep_indices) > 0:
            bboxes_raw = bboxes_raw[keep_indices]
            scores_raw = scores_raw[keep_indices]
            landmarks_raw = landmarks_raw[keep_indices]
            
            # Convert YOLO format (center_x, center_y, width, height) to (x1, y1, x2, y2)
            for bbox_raw in bboxes_raw:
                x1 = (bbox_raw[0] - bbox_raw[2] / 2) * ratio_width
                y1 = (bbox_raw[1] - bbox_raw[3] / 2) * ratio_height
                x2 = (bbox_raw[0] + bbox_raw[2] / 2) * ratio_width
                y2 = (bbox_raw[1] + bbox_raw[3] / 2) * ratio_height
                bboxes_list.append(np.array([x1, y1, x2, y2]))
            
            scores_list = scores_raw.tolist()
            
            # Convert landmarks (15 values: x1,y1,conf1,x2,y2,conf2,...x5,y5,conf5)
            # Extract only x,y coordinates (skip confidence values)
            for landmark_raw in landmarks_raw:
                # Reshape to (5, 3) then take only first 2 columns (x, y)
                landmarks_reshaped = landmark_raw.reshape(-1, 3)[:, :2]
                landmarks_reshaped[:, 0] *= ratio_width
                landmarks_reshaped[:, 1] *= ratio_height
                landmarks_list.append(landmarks_reshaped)
        
        return bboxes_list, scores_list, landmarks_list
    
    def _parse_yunet_detection(
        self,
        detection: List[NDArray],
        original_image: VisionFrame,
        score_threshold: float
    ) -> Tuple[List[BoundingBox], List[float], List[FaceLandmark5]]:
        """Parse YuNet detection outputs - match facefusion's approach."""
        bboxes_list = []
        scores_list = []
        landmarks_list = []
        
        feature_strides = [8, 16, 32]
        feature_map_channel = 3
        anchor_total = 1
        
        orig_height, orig_width = original_image.shape[:2]
        detect_width, detect_height = self.detector_size
        
        # Calculate actual scale used
        scale = min(detect_height / orig_height, detect_width / orig_width)
        temp_height = int(orig_height * scale)
        temp_width = int(orig_width * scale)
        
        ratio_height = orig_height / temp_height
        ratio_width = orig_width / temp_width
        
        for idx, feature_stride in enumerate(feature_strides):
            # YuNet has classification and objectness scores
            cls_scores = detection[idx]
            obj_scores = detection[idx + feature_map_channel]
            face_scores_raw = (cls_scores * obj_scores).reshape(-1)
            
            keep_indices = np.where(face_scores_raw >= score_threshold)[0]
            
            if len(keep_indices) > 0:
                stride_height = detect_height // feature_stride
                stride_width = detect_width // feature_stride
                anchors = self._create_anchors(feature_stride, anchor_total, stride_height, stride_width)
                
                # Get bbox predictions (center + size)
                bbox_preds = detection[idx + feature_map_channel * 2].squeeze(0)
                bboxes_center = bbox_preds[:, :2] * feature_stride + anchors
                bboxes_size = np.exp(bbox_preds[:, 2:4]) * feature_stride
                
                # Convert to x1, y1, x2, y2 format
                x1 = (bboxes_center[:, 0] - bboxes_size[:, 0] / 2) * ratio_width
                y1 = (bboxes_center[:, 1] - bboxes_size[:, 1] / 2) * ratio_height
                x2 = (bboxes_center[:, 0] + bboxes_size[:, 0] / 2) * ratio_width
                y2 = (bboxes_center[:, 1] + bboxes_size[:, 1] / 2) * ratio_height
                
                for i in keep_indices:
                    bboxes_list.append(np.array([x1[i], y1[i], x2[i], y2[i]]))
                
                scores_list.extend(face_scores_raw[keep_indices].tolist())
                
                # Get landmark predictions
                landmark_preds = detection[idx + feature_map_channel * 3].squeeze(0)
                landmarks_decoded = np.concatenate([
                    landmark_preds[:, [0, 1]] * feature_stride + anchors,
                    landmark_preds[:, [2, 3]] * feature_stride + anchors,
                    landmark_preds[:, [4, 5]] * feature_stride + anchors,
                    landmark_preds[:, [6, 7]] * feature_stride + anchors,
                    landmark_preds[:, [8, 9]] * feature_stride + anchors
                ], axis=-1).reshape(-1, 5, 2)
                
                for i in keep_indices:
                    landmarks_list.append(landmarks_decoded[i] * [ratio_width, ratio_height])
        
        return bboxes_list, scores_list, landmarks_list
    
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
    
    def _get_face_embedding(self, image: VisionFrame, face: Face) -> Optional[Tuple[NDArray, NDArray]]:
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
            embedding_norm = embedding / np.linalg.norm(embedding)
            
            return embedding, embedding_norm
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
        except Exception:
            return None


# Global detector instances - one per model
_detector_instances = {}


def get_face_detector(detector_model: str = 'scrfd') -> FaceDetector:
    """Get global face detector instance for the specified model."""
    global _detector_instances
    
    # Map user-friendly names to actual model names
    model_mapping = {
        'scrfd': 'scrfd_2.5g',
        'retinaface': 'retinaface_10g',
        'yolo_face': 'yoloface_8n',
        'yunet': 'yunet_2023_mar',
        'many': 'scrfd_2.5g'  # 'many' uses scrfd as primary, can be enhanced later
    }
    
    actual_model = model_mapping.get(detector_model, 'scrfd_2.5g')
    
    if actual_model not in _detector_instances:
        _detector_instances[actual_model] = FaceDetector(detector_model=actual_model)
    
    return _detector_instances[actual_model]


def detect_faces(
    image: VisionFrame,
    score_threshold: float = 0.3,
    sort_order: str = 'large-small',
    detector_model: str = 'scrfd'
) -> List[Face]:
    """Detect and sort faces in an image."""
    detector = get_face_detector(detector_model)
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
