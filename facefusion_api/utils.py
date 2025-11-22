"""
Utility functions for face detection, analysis, and processing.
This module provides local implementations without relying on external APIs.
"""
import os
import hashlib
from io import BytesIO
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

# Type aliases
VisionFrame = NDArray[Any]
BoundingBox = NDArray[Any]
FaceLandmark5 = NDArray[Any]
Face = Dict[str, Any]


def tensor_to_cv2(tensor: Tensor) -> VisionFrame:
    """Convert ComfyUI tensor to OpenCV image."""
    # tensor shape: [B, H, W, C] or [H, W, C]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and scale to 0-255
    img = tensor.cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img


def cv2_to_tensor(image: VisionFrame) -> Tensor:
    """Convert OpenCV image to ComfyUI tensor."""
    # Convert BGR to RGB
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to 0-1
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(image).unsqueeze(0)
    return tensor


def calculate_distance(embedding1: NDArray, embedding2: NDArray) -> float:
    """Calculate cosine distance between two face embeddings."""
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    distance = 1.0 - np.dot(embedding1_norm, embedding2_norm)
    # Normalize to 0-1 range
    distance = float(np.interp(distance, [0, 2], [0, 1]))
    return distance


def create_face_dict(
    bounding_box: BoundingBox,
    landmarks: FaceLandmark5,
    score: float,
    embedding: Optional[NDArray] = None
) -> Face:
    """Create a face dictionary with detection results."""
    return {
        'bbox': bounding_box,
        'landmarks': landmarks,
        'score': score,
        'embedding': embedding,
        'area': (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])
    }


def sort_faces_by_order(faces: List[Face], order: str = 'large-small') -> List[Face]:
    """Sort faces by specified order."""
    if not faces:
        return faces
    
    if order == 'left-right':
        return sorted(faces, key=lambda f: f['bbox'][0])
    elif order == 'right-left':
        return sorted(faces, key=lambda f: f['bbox'][0], reverse=True)
    elif order == 'top-bottom':
        return sorted(faces, key=lambda f: f['bbox'][1])
    elif order == 'bottom-top':
        return sorted(faces, key=lambda f: f['bbox'][1], reverse=True)
    elif order == 'small-large':
        return sorted(faces, key=lambda f: f['area'])
    elif order == 'large-small':
        return sorted(faces, key=lambda f: f['area'], reverse=True)
    elif order == 'best-worst':
        return sorted(faces, key=lambda f: f['score'], reverse=True)
    elif order == 'worst-best':
        return sorted(faces, key=lambda f: f['score'])
    
    return faces


def select_face_by_position(faces: List[Face], position: int = 0) -> Optional[Face]:
    """Select a face by position index."""
    if not faces:
        return None
    position = min(position, len(faces) - 1)
    return faces[position]


def find_matching_faces(
    reference_face: Face,
    target_faces: List[Face],
    distance_threshold: float = 0.6
) -> List[Face]:
    """Find faces that match the reference face based on distance threshold."""
    if not reference_face or not reference_face.get('embedding') is not None:
        return []
    
    matching_faces = []
    reference_embedding = reference_face['embedding']
    
    for target_face in target_faces:
        if target_face.get('embedding') is not None:
            distance = calculate_distance(reference_embedding, target_face['embedding'])
            if distance < distance_threshold:
                target_face['distance'] = distance
                matching_faces.append(target_face)
    
    # Sort by distance (closest first)
    matching_faces.sort(key=lambda f: f.get('distance', 1.0))
    return matching_faces


def get_average_embedding(faces: List[Face]) -> Optional[NDArray]:
    """Calculate average embedding from multiple faces."""
    if not faces:
        return None
    
    embeddings = [face['embedding'] for face in faces if face.get('embedding') is not None]
    if not embeddings:
        return None
    
    return np.mean(embeddings, axis=0)


def implode_pixel_boost(crop_frame: VisionFrame, pixel_boost_total: int, model_size: Tuple[int, int]) -> List[VisionFrame]:
    """Split frame into sub-frames for pixel boost processing."""
    pixel_boost_frame = crop_frame.reshape(
        model_size[0], pixel_boost_total,
        model_size[1], pixel_boost_total, 3
    )
    pixel_boost_frame = pixel_boost_frame.transpose(1, 3, 0, 2, 4).reshape(
        pixel_boost_total ** 2, model_size[0], model_size[1], 3
    )
    return [pixel_boost_frame[i] for i in range(pixel_boost_total ** 2)]


def explode_pixel_boost(
    temp_frames: List[VisionFrame],
    pixel_boost_total: int,
    model_size: Tuple[int, int],
    pixel_boost_size: Tuple[int, int]
) -> VisionFrame:
    """Merge sub-frames back into full frame after pixel boost processing."""
    crop_frame = np.stack(temp_frames).reshape(
        pixel_boost_total, pixel_boost_total,
        model_size[0], model_size[1], 3
    )
    crop_frame = crop_frame.transpose(2, 0, 3, 1, 4).reshape(
        pixel_boost_size[0], pixel_boost_size[1], 3
    )
    return crop_frame


def download_file(url: str, destination: str, expected_hash: Optional[str] = None) -> bool:
    """Download a file with progress and hash verification."""
    try:
        import requests
        from tqdm import tqdm
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Check if file already exists and hash matches
        if os.path.exists(destination) and expected_hash:
            if verify_file_hash(destination, expected_hash):
                print(f"File already exists and hash matches: {destination}")
                return True
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Verify hash if provided
        if expected_hash:
            if verify_file_hash(destination, expected_hash):
                print(f"Hash verification successful")
                return True
            else:
                print(f"Hash verification failed, removing file")
                os.remove(destination)
                return False
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def verify_file_hash(file_path: str, expected_hash: str) -> bool:
    """Verify file hash using SHA256."""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        file_hash = sha256_hash.hexdigest()
        return file_hash == expected_hash
    except Exception as e:
        print(f"Error verifying hash: {e}")
        return False


def get_model_path(model_name: str) -> str:
    """Get the path where a model should be stored."""
    # Store models in ComfyUI's models directory
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    return os.path.join(base_dir, model_name)


def ensure_model_exists(model_name: str, download_url: Optional[str] = None, expected_hash: Optional[str] = None) -> bool:
    """Ensure a model file exists, downloading if necessary."""
    model_path = get_model_path(model_name)
    
    # Check if model already exists
    if os.path.exists(model_path):
        if expected_hash is None or verify_file_hash(model_path, expected_hash):
            return True
    
    # Download if URL provided
    if download_url:
        print(f"Model {model_name} not found, downloading...")
        return download_file(download_url, model_path, expected_hash)
    
    return False

