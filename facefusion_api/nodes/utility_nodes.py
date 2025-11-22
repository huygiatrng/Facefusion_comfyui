"""
Utility Nodes for ComfyUI.
"""
from .base import *

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
