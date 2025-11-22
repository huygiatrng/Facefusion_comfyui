"""
Visualizer Nodes for ComfyUI.
"""
from .base import *

class FaceDataVisualizer:
	"""Node to visualize detected faces with bounding boxes."""
	
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'face_data': ('FACE_DATA',),
				'draw_landmarks':
				(
					'BOOLEAN',
					{
						'default': True
					}
				),
				'draw_bbox':
				(
					'BOOLEAN',
					{
						'default': True
					}
				)
			}
		}
	
	RETURN_TYPES = (IO.IMAGE,)
	FUNCTION = 'visualize'
	CATEGORY = 'FaceFusion'
	
	def visualize(self, face_data: Dict, draw_landmarks: bool, draw_bbox: bool) -> Tuple[Tensor]:
		"""Visualize detected faces."""
		import cv2
		import numpy as np
		
		try:
			# Get image and faces
			image_tensor = face_data.get('image')
			faces = face_data.get('faces', [])
			
			# Convert image to cv2 format
			if '_cv2_image' in face_data:
				# Use cached cv2 image if available
				image_cv2 = face_data['_cv2_image']
			else:
				# Convert from tensor
				image_cv2 = tensor_to_cv2(image_tensor)
			
			# Make a copy for drawing
			vis_image = image_cv2.copy()
			
			# Draw faces
			for i, face in enumerate(faces):
				if draw_bbox and 'bbox' in face:
					# Convert from list to numpy array if needed
					bbox = np.array(face['bbox']) if isinstance(face['bbox'], list) else face['bbox']
					bbox = bbox.astype(int)
					cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
					# Draw face number and score
					label = f"Face {i}"
					if 'score' in face:
						label += f" ({face['score']:.2f})"
					cv2.putText(vis_image, label, (bbox[0], bbox[1]-10), 
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				
				if draw_landmarks and 'landmarks' in face:
					# Convert from list to numpy array if needed
					landmarks = np.array(face['landmarks']) if isinstance(face['landmarks'], list) else face['landmarks']
					landmarks = landmarks.astype(int)
					for point in landmarks:
						cv2.circle(vis_image, tuple(point), 2, (0, 0, 255), -1)
			
			# Convert back to tensor
			output_tensor = cv2_to_tensor(vis_image)
			return (output_tensor,)
		except Exception as e:
			print(f"Error visualizing faces: {e}")
			import traceback
			traceback.print_exc()
			return (image_tensor,)




class FaceMaskVisualizer:
	"""Node to visualize face occluder and parser masks."""
	
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'face_data': ('FACE_DATA',),
				'mask_type':
				(
					['occluder', 'parser', 'combined'],
					{
						'default': 'occluder'
					}
				),
				'face_occluder_model':
				(
					['none', 'xseg_1', 'xseg_2', 'xseg_3'],
					{
						'default': 'xseg_1'
					}
				),
				'face_parser_model':
				(
					['none', 'bisenet_resnet_18', 'bisenet_resnet_34'],
					{
						'default': 'bisenet_resnet_34'
					}
				),
				'process_mode':
				(
					['single', 'all'],
					{
						'default': 'all'
					}
				),
				'face_index':
				(
					'INT',
					{
						'default': 0,
						'min': 0,
						'max': 100,
						'step': 1
					}
				),
				'visualization_mode':
				(
					['heatmap', 'overlay', 'mask_only'],
					{
						'default': 'overlay'
					}
				),
				'overlay_alpha':
				(
					'FLOAT',
					{
						'default': 0.5,
						'min': 0.0,
						'max': 1.0,
						'step': 0.1
					}
				)
			}
		}
	
	RETURN_TYPES = (IO.IMAGE,)
	FUNCTION = 'visualize_mask'
	CATEGORY = 'FaceFusion'
	
	def visualize_mask(
		self,
		face_data: Dict,
		mask_type: str,
		face_occluder_model: str,
		face_parser_model: str,
		process_mode: str,
		face_index: int,
		visualization_mode: str,
		overlay_alpha: float
	) -> Tuple[Tensor]:
		"""Visualize face masks - supports single face or all faces."""
		import cv2
		
		try:
			# Get image and faces
			image_tensor = face_data.get('image')
			faces = face_data.get('faces', [])
			
			if not faces:
				print("[FaceMaskVisualizer] No faces in face_data")
				return (image_tensor,)
			
			# Convert image to cv2 format
			if '_cv2_image' in face_data:
				image_cv2 = face_data['_cv2_image']
			else:
				image_cv2 = tensor_to_cv2(image_tensor)
			
			# Determine which faces to process
			if process_mode == 'all':
				faces_to_process = faces
				print(f"[FaceMaskVisualizer] Processing all {len(faces)} faces")
			else:  # single
				if face_index >= len(faces):
					print(f"[FaceMaskVisualizer] Face index {face_index} out of range (only {len(faces)} faces detected)")
					face_index = 0
				faces_to_process = [faces[face_index]]
			
			# Process each face
			output_images = []
			
			for idx, target_face in enumerate(faces_to_process):
				landmarks = target_face.get('landmarks')
				
				if landmarks is None:
					print(f"[FaceMaskVisualizer] No landmarks available for face {idx}")
					continue
				
				# Ensure landmarks are numpy array
				if not isinstance(landmarks, np.ndarray):
					landmarks = np.array(landmarks, dtype=np.float32)
				else:
					landmarks = landmarks.astype(np.float32)
				
				# Get a swapper instance to use alignment method
				swapper = get_local_swapper('hyperswap_1c_256')
				model_config = MODEL_CONFIGS['hyperswap_1c_256']
				crop_size = (256, 256)
				template_name = model_config['template']
				
				# Warp face to aligned crop
				crop_frame, affine_matrix = swapper._warp_face(
					image_cv2, landmarks, template_name, crop_size
				)
				
				# Create masks based on mask_type
				occluder_mask = None
				parser_mask = None
				
				if mask_type in ['occluder', 'combined'] and face_occluder_model != 'none':
					occluder = get_face_occluder(face_occluder_model)
					occluder_mask = occluder.create_occlusion_mask(crop_frame)
					if occluder_mask is not None and process_mode == 'single':
						print(f"[FaceMaskVisualizer] Occluder mask: min={occluder_mask.min():.3f}, max={occluder_mask.max():.3f}, mean={occluder_mask.mean():.3f}")
				
				if mask_type in ['parser', 'combined'] and face_parser_model != 'none':
					parser = get_face_parser(face_parser_model)
					parser_mask = parser.create_region_mask(crop_frame)
					if parser_mask is not None and process_mode == 'single':
						print(f"[FaceMaskVisualizer] Parser mask: min={parser_mask.min():.3f}, max={parser_mask.max():.3f}, mean={parser_mask.mean():.3f}")
				
				# Combine masks if needed
				if mask_type == 'combined':
					if occluder_mask is not None and parser_mask is not None:
						final_mask = occluder_mask * parser_mask
					elif occluder_mask is not None:
						final_mask = occluder_mask
					elif parser_mask is not None:
						final_mask = parser_mask
					else:
						final_mask = None
				elif mask_type == 'occluder':
					final_mask = occluder_mask
				else:  # parser
					final_mask = parser_mask
				
				if final_mask is None:
					print(f"[FaceMaskVisualizer] No mask created for face {idx}")
					continue
				
				# Create visualization
				if visualization_mode == 'mask_only':
					# Show mask as grayscale image
					mask_vis = (final_mask * 255).astype(np.uint8)
					mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
					output_cv2 = mask_vis
				
				elif visualization_mode == 'heatmap':
					# Show mask as heatmap overlay on crop
					mask_vis = (final_mask * 255).astype(np.uint8)
					heatmap = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
					output_cv2 = cv2.addWeighted(crop_frame, 1 - overlay_alpha, heatmap, overlay_alpha, 0)
				
				else:  # overlay
					# Show mask as semi-transparent overlay on crop
					mask_vis = (final_mask * 255).astype(np.uint8)
					# Create green overlay where mask is active
					overlay = crop_frame.copy()
					overlay[:, :, 1] = np.maximum(overlay[:, :, 1], mask_vis)  # Green channel
					output_cv2 = cv2.addWeighted(crop_frame, 1 - overlay_alpha, overlay, overlay_alpha, 0)
				
				# Convert to tensor
				output_tensor = cv2_to_tensor(output_cv2)
				output_images.append(output_tensor)
			
			# Return results
			if not output_images:
				print("[FaceMaskVisualizer] No masks created")
				return (image_tensor,)
			
			if len(output_images) == 1:
				# Single image - return as is
				return (output_images[0],)
			else:
				# Multiple images - stack into batch
				output_batch = torch.cat(output_images, dim=0)
				print(f"[FaceMaskVisualizer] Created batch of {len(output_images)} mask visualizations")
				return (output_batch,)
			
		except Exception as e:
			print(f"[FaceMaskVisualizer] Error: {e}")
			import traceback
			traceback.print_exc()
			return (face_data.get('image'),)
