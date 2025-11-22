from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import Tuple, Optional, List, Dict, Any

import torch
import numpy as np
from comfy.comfy_types import IO
from comfy_api.input_impl.video_types import VideoFromComponents
from comfy_api.util import VideoComponents
from comfy_api_nodes.util import bytesio_to_image_tensor, tensor_to_bytesio
from httpx import Client as HttpClient, Headers
from httpx_retries import Retry, RetryTransport
from torch import Tensor

from .types import FaceSwapperModel, InputTypes
from .utils import tensor_to_cv2, cv2_to_tensor, get_average_embedding, implode_pixel_boost, explode_pixel_boost
from .face_detector import detect_faces, select_faces
from .local_swapper import swap_faces_local

# Import content filter for NSFW detection
import sys
import os
_content_filter_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'content_filter')
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


class SwapFaceImage:
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'source_images': (IO.IMAGE,),  # Changed to plural to support batches
				'target_image': (IO.IMAGE,),
				'api_token':
				(
					'STRING',
					{
						'default': '-1'
					}
				),
				'face_swapper_model':
				(
					[
						'hyperswap_1a_256',
						'hyperswap_1b_256',
						'hyperswap_1c_256'
					],
					{
						'default': 'hyperswap_1c_256'
					}
				)
			}
		}

	RETURN_TYPES = (IO.IMAGE,)
	FUNCTION = 'process'
	CATEGORY = 'FaceFusion API'

	@staticmethod
	def process(source_images : Tensor, target_image : Tensor, api_token : str, face_swapper_model : FaceSwapperModel) -> Tuple[Tensor]:
		# Smart batch processing - handle any input format
		# Use first source image (or average multiple sources in future)
		if source_images.dim() == 4 and source_images.shape[0] > 1:
			source_image = source_images[0:1]
		else:
			source_image = source_images
		
		# Check if target is a batch
		if target_image.dim() == 4 and target_image.shape[0] > 1:
			# Process each target image in the batch
			print(f"[SwapFaceImage] Processing batch of {target_image.shape[0]} images")
			output_images = []
			for i in range(target_image.shape[0]):
				single_target = target_image[i:i+1]
				swapped = SwapFaceImage.swap_face(source_image, single_target, api_token, face_swapper_model, '512x512', 0.3)
				output_images.append(swapped)
			# Stack all results back into batch
			output_tensor = torch.cat(output_images, dim=0)
		else:
			# Single image processing
			output_tensor = SwapFaceImage.swap_face(source_image, target_image, api_token, face_swapper_model, '512x512', 0.3)
		
		return (output_tensor,)

	@staticmethod
	def swap_face(source_tensor : Tensor, target_tensor : Tensor, api_token : str, face_swapper_model : FaceSwapperModel, pixel_boost: str = '512x512', face_mask_blur: float = 0.3, face_occluder_model: Optional[str] = None, face_parser_model: Optional[str] = None) -> Tensor:
		# Check if using local inference
		if api_token == '-1':
			# print("[SwapFaceImage] Using local inference")
			try:
				# Convert tensors to OpenCV format
				source_cv2 = tensor_to_cv2(source_tensor)
				target_cv2 = tensor_to_cv2(target_tensor)
				
				# NSFW content detection
				is_source_nsfw = analyse_frame(source_cv2)
				is_target_nsfw = analyse_frame(target_cv2)
				
				if is_source_nsfw or is_target_nsfw:
					print("[ContentFilter] NSFW content detected - returning blurred output")
					# Return blurred version of target
					blurred = blur_frame(target_cv2)
					return cv2_to_tensor(blurred)
				
				# Perform local face swap
				result_cv2 = swap_faces_local(
					source_image=source_cv2,
					target_image=target_cv2,
					model_name=face_swapper_model,
					pixel_boost=pixel_boost,
					face_mask_blur=face_mask_blur,
					face_selector_mode='one',
					face_position=0,
					score_threshold=0.3,
					face_occluder_model=face_occluder_model,
					face_parser_model=face_parser_model
				)
				
				# Convert back to tensor
				result_tensor = cv2_to_tensor(result_cv2)
				return result_tensor
			except Exception as e:
				print(f"[SwapFaceImage] Local inference error: {e}")
				import traceback
				traceback.print_exc()
				return target_tensor
		
		# Use API
		# print("[SwapFaceImage] Using API inference")
		source_buffer : BytesIO = tensor_to_bytesio(source_tensor, mime_type = 'image/webp')
		target_buffer : BytesIO = tensor_to_bytesio(target_tensor, mime_type = 'image/webp')

		url = 'https://api.facefusion.io/inferences/swap-face'
		files =\
		{
			'source': ('source.webp', source_buffer, 'image/webp'),
			'target': ('target.webp', target_buffer, 'image/webp'),
		}
		data =\
		{
			'face_swapper_model': face_swapper_model,
		}
		headers = Headers()
		retry = Retry(total = 5, backoff_factor = 1)
		transport = RetryTransport(retry = retry)

		if api_token and api_token != '-1':
			headers['X-Token'] = api_token

		with HttpClient(transport = transport) as http_client:
			response = http_client.post(url, headers = headers, files = files, data = data)

			if response.status_code == 200:
				output_buffer = BytesIO(response.content)
				output_tensor = bytesio_to_image_tensor(output_buffer)
				return output_tensor

		return target_tensor


class SwapFaceVideo:
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'source_images': (IO.IMAGE,),  # Changed to plural to support batches
				'target_video': (IO.VIDEO,),
				'api_token':
				(
					'STRING',
					{
						'default': '-1'
					}
				),
				'face_swapper_model':
				(
					[
						'hyperswap_1a_256',
						'hyperswap_1b_256',
						'hyperswap_1c_256'
					],
					{
						'default': 'hyperswap_1a_256'
					}
				),
				'max_workers':
				(
					'INT',
					{
						'default': 16,
						'min': 1,
						'max': 32
					}
				)
			}
		}

	RETURN_TYPES = (IO.VIDEO,)
	FUNCTION = 'process'
	CATEGORY = 'FaceFusion API'

	@staticmethod
	def process(source_images : Tensor, target_video : VideoFromComponents, api_token : str, face_swapper_model : FaceSwapperModel, max_workers : int) -> Tuple[VideoFromComponents]:
		try:
			# Handle multiple source images by taking the first one
			if source_images.dim() == 4 and source_images.shape[0] > 1:
				source_image = source_images[0:1]
			else:
				source_image = source_images
			
			# Get video components with error handling
			try:
				video_components = target_video.get_components()
			except Exception as e:
				error_msg = str(e)
				if 'Invalid data found' in error_msg or 'avcodec' in error_msg or 'Number of bands' in error_msg:
					print(f"[SwapFaceVideo] ❌ Video has corrupted audio codec")
					print(f"[SwapFaceVideo] Error: {error_msg}")
					print("[SwapFaceVideo] ")
					print("[SwapFaceVideo] 🔧 To fix this video, re-encode with:")
					print("[SwapFaceVideo]    ffmpeg -i input.mp4 -c:v copy -c:a aac -b:a 128k -ar 44100 output.mp4")
					print("[SwapFaceVideo] ")
					raise RuntimeError(f"Cannot process video with corrupted audio. Please re-encode the video first.")
				else:
					raise
			
			# Check source image for NSFW content (only if using local inference)
			if api_token == '-1' and CONTENT_FILTER_AVAILABLE:
				source_cv2 = tensor_to_cv2(source_image)
				if analyse_frame(source_cv2):
					print("[ContentFilter] NSFW source detected in video - returning blurred video")
					# Blur all frames
					blurred_frames = []
					for frame_tensor in video_components.images:
						frame_cv2 = tensor_to_cv2(frame_tensor.unsqueeze(0))
						blurred_cv2 = blur_frame(frame_cv2)
						blurred_tensor = cv2_to_tensor(blurred_cv2).squeeze(0)[..., :3]
						blurred_frames.append(blurred_tensor)
					
					output_video_components = VideoComponents(
						images = torch.stack(blurred_frames),
						audio = video_components.audio,
						frame_rate = video_components.frame_rate
					)
					return (VideoFromComponents(output_video_components),)
			
			# Sample check for NSFW in target video (check first, middle, last frame)
			if api_token == '-1' and CONTENT_FILTER_AVAILABLE and len(video_components.images) > 0:
				sample_indices = [0, len(video_components.images) // 2, len(video_components.images) - 1]
				nsfw_detected = False
				
				for idx in sample_indices:
					if idx < len(video_components.images):
						frame_tensor = video_components.images[idx].unsqueeze(0)
						frame_cv2 = tensor_to_cv2(frame_tensor)
						if analyse_frame(frame_cv2):
							nsfw_detected = True
							break
				
				if nsfw_detected:
					print("[ContentFilter] NSFW content detected in target video - returning blurred video")
					blurred_frames = []
					for frame_tensor in video_components.images:
						frame_cv2 = tensor_to_cv2(frame_tensor.unsqueeze(0))
						blurred_cv2 = blur_frame(frame_cv2)
						blurred_tensor = cv2_to_tensor(blurred_cv2).squeeze(0)[..., :3]
						blurred_frames.append(blurred_tensor)
					
					output_video_components = VideoComponents(
						images = torch.stack(blurred_frames),
						audio = video_components.audio,
						frame_rate = video_components.frame_rate
					)
					return (VideoFromComponents(output_video_components),)
			
			output_tensors = []

			swap_face = partial(
				SwapFaceImage.swap_face,
				source_image,
				api_token = api_token,
				face_swapper_model = face_swapper_model
			)

			with ThreadPoolExecutor(max_workers = max_workers) as executor:
				for temp_tensor in executor.map(swap_face, video_components.images):
					temp_tensor = temp_tensor.squeeze(0)[..., :3]
					output_tensors.append(temp_tensor)

			output_video_components = VideoComponents(
				images = torch.stack(output_tensors),
				audio = video_components.audio,
				frame_rate = video_components.frame_rate
			)

			output_video = VideoFromComponents(output_video_components)
			return (output_video,)
			
		except RuntimeError as e:
			# Re-raise RuntimeError with clear message (don't return original video)
			print(f"[SwapFaceVideo] Fatal error: {e}")
			raise
		except Exception as e:
			print(f"[SwapFaceVideo] Unexpected error: {e}")
			import traceback
			traceback.print_exc()
			# Return original video on unexpected errors only
			return (target_video,)


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
			
			# Note: face_detector_model selection will be implemented when multi-model support is added
			# Currently only SCRFD is implemented
			if face_detector_model != 'scrfd':
				print(f"[FaceDetectorNode] Warning: Only 'scrfd' is currently implemented. Using SCRFD.")
			
			# Detect faces with sorting
			faces = detect_faces(cv2_image, score_threshold, sort_order)
			
			if not faces:
				print("No faces detected in image")
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


class AdvancedSwapFaceImage:
	"""Advanced face swapping node with face selection options."""
	
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'source_images': (IO.IMAGE,),
				'target_image': (IO.IMAGE,),
				'api_token':
				(
					'STRING',
					{
						'default': '-1'
					}
				),
				'face_swapper_model':
				(
					[
						'hyperswap_1a_256',
						'hyperswap_1b_256',
						'hyperswap_1c_256'
					],
					{
						'default': 'hyperswap_1c_256'
					}
				),
				'pixel_boost':
				(
					['256x256', '512x512', '768x768', '1024x1024'],
					{
						'default': '512x512'
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
				'face_mask_blur':
				(
					'FLOAT',
					{
						'default': 0.3,
						'min': 0.0,
						'max': 1.0,
						'step': 0.05
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
						'max': 100
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
	
	RETURN_TYPES = (IO.IMAGE,)
	FUNCTION = 'process'
	CATEGORY = 'FaceFusion'
	
	def process(
		self,
		source_images: Tensor,
		target_image: Tensor,
		api_token: str,
		face_swapper_model: FaceSwapperModel,
		pixel_boost: str,
		face_occluder_model: str,
		face_parser_model: str,
		face_mask_blur: float,
		face_selector_mode: str,
		face_position: int,
		score_threshold: float,
		reference_image: Optional[Tensor] = None,
		reference_face_distance: float = 0.6
	) -> Tuple[Tensor]:
		"""Process face swapping with advanced selection - smart batch handling."""
		# Note: pixel_boost, face_occluder, and face_parser are used with local inference
		# print(f"[AdvancedSwapFaceImage] Settings: pixel_boost={pixel_boost}, occluder={face_occluder_model}, parser={face_parser_model}")
		
		# Handle multiple source images - use first one
		if source_images.dim() == 4 and source_images.shape[0] > 1:
			source_image = source_images[0:1]
		else:
			source_image = source_images
		
		# Smart batch processing for target images
		if target_image.dim() == 4 and target_image.shape[0] > 1:
			# Process batch of target images
			batch_size = target_image.shape[0]
			print(f"[AdvancedSwapFaceImage] Processing batch of {batch_size} images")
			output_images = []
			
			for i in range(batch_size):
				single_target = target_image[i:i+1]
				swapped = SwapFaceImage.swap_face(
					source_image, 
					single_target, 
					api_token, 
					face_swapper_model, 
					pixel_boost, 
					face_mask_blur,
					face_occluder_model,
					face_parser_model
				)
				output_images.append(swapped)
			
			# Stack results maintaining batch format
			output_tensor = torch.cat(output_images, dim=0)
		else:
			# Single image processing
			output_tensor = SwapFaceImage.swap_face(
				source_image, 
				target_image, 
				api_token, 
				face_swapper_model, 
				pixel_boost, 
				face_mask_blur,
				face_occluder_model,
				face_parser_model
			)
		
		return (output_tensor,)


class AdvancedSwapFaceVideo:
	"""Advanced video face swapping node with face selection options."""
	
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'source_images': (IO.IMAGE,),
				'target_video': (IO.VIDEO,),
				'api_token':
				(
					'STRING',
					{
						'default': '-1'
					}
				),
				'face_swapper_model':
				(
					[
						'hyperswap_1a_256',
						'hyperswap_1b_256',
						'hyperswap_1c_256'
					],
					{
						'default': 'hyperswap_1a_256'
					}
				),
				'pixel_boost':
				(
					['256x256', '512x512', '768x768', '1024x1024'],
					{
						'default': '512x512'
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
				'face_mask_blur':
				(
					'FLOAT',
					{
						'default': 0.3,
						'min': 0.0,
						'max': 1.0,
						'step': 0.05
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
						'max': 100
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
				),
				'max_workers':
				(
					'INT',
					{
						'default': 16,
						'min': 1,
						'max': 32
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
	
	RETURN_TYPES = (IO.VIDEO,)
	FUNCTION = 'process'
	CATEGORY = 'FaceFusion'
	
	def process(
		self,
		source_images: Tensor,
		target_video: VideoFromComponents,
		api_token: str,
		face_swapper_model: FaceSwapperModel,
		pixel_boost: str,
		face_occluder_model: str,
		face_parser_model: str,
		face_mask_blur: float,
		face_selector_mode: str,
		face_position: int,
		score_threshold: float,
		max_workers: int,
		reference_image: Optional[Tensor] = None,
		reference_face_distance: float = 0.6
	) -> Tuple[VideoFromComponents]:
		"""Process video face swapping with advanced selection."""
		try:
			# Handle multiple source images
			if source_images.dim() == 4 and source_images.shape[0] > 1:
				source_image = source_images[0:1]
			else:
				source_image = source_images
			
			# Get video components with error handling
			try:
				video_components = target_video.get_components()
			except Exception as e:
				error_msg = str(e)
				if 'Invalid data found' in error_msg or 'avcodec' in error_msg or 'Number of bands' in error_msg:
					print(f"[AdvancedSwapFaceVideo] ❌ Video has corrupted audio codec")
					print(f"[AdvancedSwapFaceVideo] Error: {error_msg}")
					print("[AdvancedSwapFaceVideo] ")
					print("[AdvancedSwapFaceVideo] 🔧 To fix this video, re-encode with:")
					print("[AdvancedSwapFaceVideo]    ffmpeg -i input.mp4 -c:v copy -c:a aac -b:a 128k -ar 44100 output.mp4")
					print("[AdvancedSwapFaceVideo] ")
					raise RuntimeError(f"Cannot process video with corrupted audio. Please re-encode the video first.")
				else:
					raise
			
			# Check source image for NSFW content (only if using local inference)
			if api_token == '-1' and CONTENT_FILTER_AVAILABLE:
				source_cv2 = tensor_to_cv2(source_image)
				if analyse_frame(source_cv2):
					print("[ContentFilter] NSFW source detected in video - returning blurred video")
					blurred_frames = []
					for frame_tensor in video_components.images:
						frame_cv2 = tensor_to_cv2(frame_tensor.unsqueeze(0))
						blurred_cv2 = blur_frame(frame_cv2)
						blurred_tensor = cv2_to_tensor(blurred_cv2).squeeze(0)[..., :3]
						blurred_frames.append(blurred_tensor)
					
					output_video_components = VideoComponents(
						images = torch.stack(blurred_frames),
						audio = video_components.audio,
						frame_rate = video_components.frame_rate
					)
					return (VideoFromComponents(output_video_components),)
			
			# Sample check for NSFW in target video (check first, middle, last frame)
			if api_token == '-1' and CONTENT_FILTER_AVAILABLE and len(video_components.images) > 0:
				sample_indices = [0, len(video_components.images) // 2, len(video_components.images) - 1]
				nsfw_detected = False
				
				for idx in sample_indices:
					if idx < len(video_components.images):
						frame_tensor = video_components.images[idx].unsqueeze(0)
						frame_cv2 = tensor_to_cv2(frame_tensor)
						if analyse_frame(frame_cv2):
							nsfw_detected = True
							break
				
				if nsfw_detected:
					print("[ContentFilter] NSFW content detected in target video - returning blurred video")
					blurred_frames = []
					for frame_tensor in video_components.images:
						frame_cv2 = tensor_to_cv2(frame_tensor.unsqueeze(0))
						blurred_cv2 = blur_frame(frame_cv2)
						blurred_tensor = cv2_to_tensor(blurred_cv2).squeeze(0)[..., :3]
						blurred_frames.append(blurred_tensor)
					
					output_video_components = VideoComponents(
						images = torch.stack(blurred_frames),
						audio = video_components.audio,
						frame_rate = video_components.frame_rate
					)
					return (VideoFromComponents(output_video_components),)
			
			output_tensors = []

			swap_face = partial(
				SwapFaceImage.swap_face,
				source_image,
				api_token = api_token,
				face_swapper_model = face_swapper_model,
				pixel_boost = pixel_boost,
				face_mask_blur = face_mask_blur
			)

			with ThreadPoolExecutor(max_workers = max_workers) as executor:
				for temp_tensor in executor.map(swap_face, video_components.images):
					temp_tensor = temp_tensor.squeeze(0)[..., :3]
					output_tensors.append(temp_tensor)

			output_video_components = VideoComponents(
				images = torch.stack(output_tensors),
				audio = video_components.audio,
				frame_rate = video_components.frame_rate
			)

			output_video = VideoFromComponents(output_video_components)
			return (output_video,)
			
		except RuntimeError as e:
			# Re-raise RuntimeError with clear message (don't return original video)
			print(f"[AdvancedSwapFaceVideo] Fatal error: {e}")
			raise
		except Exception as e:
			print(f"[AdvancedSwapFaceVideo] Unexpected error: {e}")
			import traceback
			traceback.print_exc()
			# Return original video on unexpected errors only
			return (target_video,)


class PixelBoostNode:
	"""Node for setting pixel boost resolution (for local face swapping)."""
	
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'image': (IO.IMAGE,),
				'pixel_boost':
				(
					['256x256', '512x512', '768x768', '1024x1024'],
					{
						'default': '512x512'
					}
				)
			}
		}
	
	RETURN_TYPES = (IO.IMAGE, 'STRING')
	RETURN_NAMES = ('image', 'pixel_boost_setting')
	FUNCTION = 'process'
	CATEGORY = 'FaceFusion'
	
	def process(self, image: Tensor, pixel_boost: str) -> Tuple[Tensor, str]:
		"""Pass through image and pixel boost setting."""
		# This node serves as a configuration node for pixel boost settings
		# The actual pixel boost processing happens in the face swapping nodes
		# For API-based workflow, this setting is informational only
		# print(f"[PixelBoostNode] Setting: {pixel_boost}")
		# print(f"[PixelBoostNode] Note: Full pixel boost will be available with local face swapping")
		return (image, pixel_boost)


class FaceSwapApplier:
	"""Node to apply face swap to specific detected faces."""
	
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'source_images': (IO.IMAGE,),
				'target_face_data': ('FACE_DATA',),
				'api_token':
				(
					'STRING',
					{
						'default': '-1'
					}
				),
				'face_swapper_model':
				(
					[
						'hyperswap_1a_256',
						'hyperswap_1b_256',
						'hyperswap_1c_256'
					],
					{
						'default': 'hyperswap_1c_256'
					}
				),
				'pixel_boost':
				(
					['256x256', '512x512', '768x768', '1024x1024'],
					{
						'default': '512x512'
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
				'face_mask_blur':
				(
					'FLOAT',
					{
						'default': 0.3,
						'min': 0.0,
						'max': 1.0,
						'step': 0.05
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
				)
			}
		}
	
	RETURN_TYPES = (IO.IMAGE, 'FACE_DATA')
	RETURN_NAMES = ('swapped_image', 'face_data')
	FUNCTION = 'apply'
	CATEGORY = 'FaceFusion'
	
	def apply(
		self,
		source_images: Tensor,
		target_face_data: Dict,
		api_token: str,
		face_swapper_model: FaceSwapperModel,
		pixel_boost: str,
		face_occluder_model: str,
		face_parser_model: str,
		face_mask_blur: float,
		face_index: int
	) -> Tuple[Tensor, Dict]:
		"""Apply face swap to specific detected face - smart batch handling."""
		try:
			# Get target image
			target_image = target_face_data.get('image')
			faces = target_face_data.get('faces', [])
			
			if not faces:
				print("No faces in face_data to swap")
				return (target_image, target_face_data)
			
			if face_index >= len(faces):
				print(f"Face index {face_index} out of range (only {len(faces)} faces detected)")
				face_index = 0
			
			# Handle multiple source images
			if source_images.dim() == 4 and source_images.shape[0] > 1:
				source_image = source_images[0:1]
			else:
				source_image = source_images
			
			# Smart batch handling for target images
			if target_image.dim() == 4 and target_image.shape[0] > 1:
				# Process batch
				print(f"[FaceSwapApplier] Processing batch of {target_image.shape[0]} images")
				output_images = []
				for i in range(target_image.shape[0]):
					single_target = target_image[i:i+1]
					swapped = SwapFaceImage.swap_face(
						source_image, 
						single_target, 
						api_token, 
						face_swapper_model, 
						pixel_boost, 
						face_mask_blur,
						face_occluder_model,
						face_parser_model
					)
					output_images.append(swapped)
				swapped_image = torch.cat(output_images, dim=0)
			else:
				# Single image
				swapped_image = SwapFaceImage.swap_face(
					source_image, 
					target_image, 
					api_token, 
					face_swapper_model, 
					pixel_boost, 
					face_mask_blur,
					face_occluder_model,
					face_parser_model
				)
			
			print(f"Applied face swap to face {face_index}")
			
			return (swapped_image, target_face_data)
		except Exception as e:
			print(f"Error applying face swap: {e}")
			import traceback
			traceback.print_exc()
			return (target_face_data.get('image'), target_face_data)


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
		face_index: int,
		visualization_mode: str,
		overlay_alpha: float
	) -> Tuple[Tensor]:
		"""Visualize face masks."""
		import cv2
		from .local_swapper import get_face_occluder, get_face_parser, get_local_swapper, MODEL_CONFIGS
		
		try:
			# Get image and faces
			image_tensor = face_data.get('image')
			faces = face_data.get('faces', [])
			
			if not faces:
				print("[FaceMaskVisualizer] No faces in face_data")
				return (image_tensor,)
			
			if face_index >= len(faces):
				print(f"[FaceMaskVisualizer] Face index {face_index} out of range (only {len(faces)} faces detected)")
				face_index = 0
			
			# Convert image to cv2 format
			if '_cv2_image' in face_data:
				image_cv2 = face_data['_cv2_image']
			else:
				image_cv2 = tensor_to_cv2(image_tensor)
			
			# Get target face
			target_face = faces[face_index]
			landmarks = target_face.get('landmarks')
			
			if landmarks is None:
				print("[FaceMaskVisualizer] No landmarks available for face")
				return (image_tensor,)
			
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
				if occluder_mask is not None:
					print(f"[FaceMaskVisualizer] Occluder mask: min={occluder_mask.min():.3f}, max={occluder_mask.max():.3f}, mean={occluder_mask.mean():.3f}")
			
			if mask_type in ['parser', 'combined'] and face_parser_model != 'none':
				parser = get_face_parser(face_parser_model)
				parser_mask = parser.create_region_mask(crop_frame)
				if parser_mask is not None:
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
				print("[FaceMaskVisualizer] No mask created")
				return (image_tensor,)
			
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
			
			# Convert back to tensor
			output_tensor = cv2_to_tensor(output_cv2)
			return (output_tensor,)
			
		except Exception as e:
			print(f"[FaceMaskVisualizer] Error: {e}")
			import traceback
			traceback.print_exc()
			return (face_data.get('image'),)
