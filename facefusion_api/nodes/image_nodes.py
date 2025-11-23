"""
Image Nodes for ComfyUI.
"""
from .base import *

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
						'default': 'hyperswap_1c_256'
					}
				),
				'face_detector_model':
				(
					['scrfd', 'retinaface', 'yolo_face', 'yunet', 'many'],
					{
						'default': 'scrfd'
					}
				)
			}
		}

	RETURN_TYPES = (IO.IMAGE,)
	FUNCTION = 'process'
	CATEGORY = 'FaceFusion API'

	@staticmethod
	def process(source_images : Tensor, target_image : Tensor, api_token : str, face_swapper_model : FaceSwapperModel, face_detector_model: str) -> Tuple[Tensor]:
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
				swapped = SwapFaceImage.swap_face(source_image, single_target, api_token, face_swapper_model, '512x512', 0.3, face_detector_model=face_detector_model)
				output_images.append(swapped)
			# Stack all results back into batch
			output_tensor = torch.cat(output_images, dim=0)
		else:
			# Single image processing
			output_tensor = SwapFaceImage.swap_face(source_image, target_image, api_token, face_swapper_model, '512x512', 0.3, face_detector_model=face_detector_model)
		
		return (output_tensor,)

	@staticmethod
	def swap_face(source_tensor : Tensor, target_tensor : Tensor, api_token : str, face_swapper_model : FaceSwapperModel, pixel_boost: str = '512x512', face_mask_blur: float = 0.3, face_occluder_model: Optional[str] = None, face_parser_model: Optional[str] = None, face_selector_mode: str = 'one', face_position: int = 0, sort_order: str = 'large-small', score_threshold: float = 0.3, face_detector_model: str = 'scrfd') -> Tensor:
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
					face_selector_mode=face_selector_mode,
					face_position=face_position,
					sort_order=sort_order,
					score_threshold=score_threshold,
					face_occluder_model=face_occluder_model,
					face_parser_model=face_parser_model,
					face_detector_model=face_detector_model
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
						'default': 'hyperswap_1c_256'
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
		face_detector_model: str,
		pixel_boost: str,
		face_occluder_model: str,
		face_parser_model: str,
		face_mask_blur: float,
		face_selector_mode: str,
		face_position: int,
		sort_order: str,
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
					face_parser_model,
					face_selector_mode,
					face_position,
					sort_order,
					score_threshold,
					face_detector_model
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
				face_parser_model,
				face_selector_mode,
				face_position,
				sort_order,
				score_threshold,
				face_detector_model
			)
		
		return (output_tensor,)
