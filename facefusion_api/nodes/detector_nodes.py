"""
Detector Nodes for ComfyUI.
"""
from .base import *

class FaceDetectorNode:
	"""Node for detecting and selecting faces from an image."""
	
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'image': (IO.IMAGE,),
				'face_detector_model':
				(
					['scrfd', 'retinaface', 'yolo_face', 'yunet', 'many'],
					{
						'default': 'scrfd'
					}
				),
				'face_selector_mode':
				(
					['one', 'many', 'reference'],
					{
						'default': 'one'
					}
				),
				'face_position':
				(
					'INT',
					{
						'default': 0,
						'min': 0,
						'max': 100,
						'step': 1
					}
				),
				'sort_order':
				(
					['large-small', 'small-large', 'left-right', 'right-left', 'top-bottom', 'bottom-top', 'best-worst', 'worst-best'],
					{
						'default': 'large-small'
					}
				),
				'score_threshold':
				(
					'FLOAT',
					{
					'default': 0.3,
						'min': 0.0,
						'max': 1.0,
						'step': 0.05
					}
				)
			},
			'optional':
			{
				'reference_image': (IO.IMAGE,),
				'reference_face_distance':
				(
					'FLOAT',
					{
						'default': 0.6,
						'min': 0.0,
						'max': 1.0,
						'step': 0.05
					}
				)
			}
		}
	
	RETURN_TYPES = ('FACE_DATA',)
	RETURN_NAMES = ('face_data',)
	FUNCTION = 'detect'
	CATEGORY = 'FaceFusion'
	
	def detect(
		self,
		image: Tensor,
		face_detector_model: str,
		face_selector_mode: str,
		face_position: int,
		sort_order: str,
		score_threshold: float,
		reference_image: Optional[Tensor] = None,
		reference_face_distance: float = 0.6
	) -> Tuple[Dict]:
		"""Detect and select faces from an image - smart batch handling."""
		try:
			# print(f"[FaceDetectorNode] Using detector model: {face_detector_model}")
			
			# Check if input is a batch
			if image.dim() == 4 and image.shape[0] > 1:
				print(f"[FaceDetectorNode] Processing batch of {image.shape[0]} images - using first image")
				# For face detection, use first image in batch
				# (detecting faces from multiple images doesn't make sense in this context)
				single_image = image[0:1]
			else:
				single_image = image
			
			# Convert tensor to OpenCV format
			cv2_image = tensor_to_cv2(single_image)
			
			# Detect faces with sorting and specified detector model
			faces = detect_faces(cv2_image, score_threshold, sort_order, face_detector_model)
			
			if not faces:
				# print("No faces detected in image")
				# Return serializable data
				return ({'faces': [], 'image': single_image, 'num_faces': 0},)
			
			# Select faces based on mode
			selected_faces = []
			
			if face_selector_mode == 'one':
				if face_position < len(faces):
					selected_faces = [faces[face_position]]
			elif face_selector_mode == 'many':
				selected_faces = faces
			elif face_selector_mode == 'reference' and reference_image is not None:
				# Get reference face
				ref_single = reference_image[0:1] if reference_image.dim() == 4 and reference_image.shape[0] > 1 else reference_image
				ref_cv2_image = tensor_to_cv2(ref_single)
				ref_faces = detect_faces(ref_cv2_image, score_threshold, 'large-small')
				
				if ref_faces:
					# Get first reference face
					ref_face = ref_faces[0]
					# Find matching faces
					from .utils import find_matching_faces
					selected_faces = find_matching_faces(ref_face, faces, reference_face_distance)
			
			# print(f"Detected {len(faces)} faces, selected {len(selected_faces)} faces")
			
			# Convert numpy arrays to serializable format and store cv2_image separately
			serializable_faces = []
			for face in selected_faces:
				serializable_face = {
					'bbox': face['bbox'].tolist() if hasattr(face['bbox'], 'tolist') else face['bbox'],
					'landmarks': face['landmarks'].tolist() if hasattr(face['landmarks'], 'tolist') else face['landmarks'],
					'score': float(face['score']),
					'area': float(face['area'])
				}
				if 'embedding' in face and face['embedding'] is not None:
					serializable_face['embedding'] = face['embedding'].tolist()
				if 'distance' in face:
					serializable_face['distance'] = float(face['distance'])
				serializable_faces.append(serializable_face)
			
			# Store cv2_image for internal use but don't serialize it
			face_data = {
				'faces': serializable_faces,
				'image': single_image,
				'num_faces': len(selected_faces),
				'_cv2_image': cv2_image  # Internal use only
			}
			
			return (face_data,)
		except Exception as e:
			print(f"Error in face detection: {e}")
			import traceback
			traceback.print_exc()
			return ({'faces': [], 'image': image, 'num_faces': 0},)
