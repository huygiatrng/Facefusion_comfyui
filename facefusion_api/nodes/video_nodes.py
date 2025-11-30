"""
Video Nodes for ComfyUI.
"""
from .base import *
from .image_nodes import SwapFaceImage

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
						'hyperswap_1c_256',
						'ghost_1_256',
						'ghost_2_256',
						'ghost_3_256',
						'hififace_unofficial_256',
						'inswapper_128',
						'inswapper_128_fp16',
						'blendswap_256',
						'simswap_256',
						'simswap_unofficial_512',
						'uniface_256'
					],
					{
						'default': 'hyperswap_1a_256'
					}
				),
				'face_detector_model':
				(
					['scrfd', 'retinaface', 'yolo_face', 'yunet', 'many'],
					{
						'default': 'scrfd'
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
	def process(source_images : Tensor, target_video : VideoFromComponents, api_token : str, face_swapper_model : FaceSwapperModel, face_detector_model: str, max_workers : int) -> Tuple[VideoFromComponents]:
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
				face_swapper_model = face_swapper_model,
				face_detector_model = face_detector_model
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
						'hyperswap_1c_256',
						'ghost_1_256',
						'ghost_2_256',
						'ghost_3_256',
						'hififace_unofficial_256',
						'inswapper_128',
						'inswapper_128_fp16',
						'blendswap_256',
						'simswap_256',
						'simswap_unofficial_512',
						'uniface_256'
					],
					{
						'default': 'hyperswap_1a_256'
					}
				),
				'face_detector_model':
				(
					['scrfd', 'retinaface', 'yolo_face', 'yunet', 'many'],
					{
						'default': 'scrfd'
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
				),
				'use_box_mask':
				(
					'BOOLEAN',
					{
						'default': True
					}
				),
				'use_occlusion_mask':
				(
					'BOOLEAN',
					{
						'default': False
					}
				),
				'use_area_mask':
				(
					'BOOLEAN',
					{
						'default': False
					}
				),
				'use_region_mask':
				(
					'BOOLEAN',
					{
						'default': False
					}
				),
				'face_mask_areas':
				(
					'STRING',
					{
						'default': 'upper-face,lower-face,mouth',
						'multiline': False
					}
				),
				'face_mask_regions':
				(
					'STRING',
					{
						'default': 'skin,nose,mouth,upper-lip,lower-lip',
						'multiline': False
					}
				),
				'face_mask_padding':
				(
					'STRING',
					{
						'default': '0,0,0,0',
						'multiline': False
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
		face_detector_model: str,
		pixel_boost: str,
		face_occluder_model: str,
		face_parser_model: str,
		face_mask_blur: float,
		face_selector_mode: str,
		face_position: int,
		sort_order: str,
		score_threshold: float,
		use_box_mask: bool = True,
		use_occlusion_mask: bool = False,
		use_area_mask: bool = False,
		use_region_mask: bool = False,
		face_mask_areas: str = 'upper-face,lower-face,mouth',
		face_mask_regions: str = 'skin,nose,mouth,upper-lip,lower-lip',
		face_mask_padding: str = '0,0,0,0',
		max_workers: int = 16,
		reference_image: Optional[Tensor] = None,
		reference_face_distance: float = 0.6
	) -> Tuple[VideoFromComponents]:
		"""Process video face swapping with advanced selection."""
		# Build face_mask_types list based on boolean options
		face_mask_types = []
		if use_box_mask:
			face_mask_types.append('box')
		if use_occlusion_mask:
			face_mask_types.append('occlusion')
		if use_area_mask:
			face_mask_types.append('area')
		if use_region_mask:
			face_mask_types.append('region')
		
		# Parse mask areas and regions from comma-separated strings
		mask_areas = [a.strip() for a in face_mask_areas.split(',') if a.strip()]
		mask_regions = [r.strip() for r in face_mask_regions.split(',') if r.strip()]
		
		# Parse padding (top, right, bottom, left)
		try:
			padding = tuple(int(p.strip()) for p in face_mask_padding.split(','))
			if len(padding) != 4:
				padding = (0, 0, 0, 0)
		except:
			padding = (0, 0, 0, 0)
		
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
				face_mask_blur = face_mask_blur,
				face_occluder_model = face_occluder_model,
				face_parser_model = face_parser_model,
				face_selector_mode = face_selector_mode,
				face_position = face_position,
				sort_order = sort_order,
				score_threshold = score_threshold,
				face_detector_model = face_detector_model,
				face_mask_types = face_mask_types,
				face_mask_areas = mask_areas,
				face_mask_regions = mask_regions,
				face_mask_padding = padding
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
