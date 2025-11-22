from .core import (
	SwapFaceImage, SwapFaceVideo,
	AdvancedSwapFaceImage, AdvancedSwapFaceVideo,
	FaceDetectorNode, FaceSwapApplier, PixelBoostNode, FaceDataVisualizer, FaceMaskVisualizer
)
from .types import NodeClassMapping, NodeDisplayNameMapping

NODE_CLASS_MAPPINGS : NodeClassMapping =\
{
	# Basic API nodes
	'SwapFaceImage': SwapFaceImage,
	'SwapFaceVideo': SwapFaceVideo,
	# Advanced nodes
	'AdvancedSwapFaceImage': AdvancedSwapFaceImage,
	'AdvancedSwapFaceVideo': AdvancedSwapFaceVideo,
	# Face detection and utilities
	'FaceDetectorNode': FaceDetectorNode,
	'FaceSwapApplier': FaceSwapApplier,
	'PixelBoostNode': PixelBoostNode,
	'FaceDataVisualizer': FaceDataVisualizer,
	'FaceMaskVisualizer': FaceMaskVisualizer
}

NODE_DISPLAY_NAME_MAPPINGS : NodeDisplayNameMapping =\
{
	# Basic API nodes
	'SwapFaceImage': 'FF API: Swap Face (Image)',
	'SwapFaceVideo': 'FF API: Swap Face (Video)',
	# Advanced nodes
	'AdvancedSwapFaceImage': 'FF: Advanced Swap Face (Image)',
	'AdvancedSwapFaceVideo': 'FF: Advanced Swap Face (Video)',
	# Face detection and utilities
	'FaceDetectorNode': 'FF: Face Detector',
	'FaceSwapApplier': 'FF: Face Swap Applier',
	'PixelBoostNode': 'FF: Pixel Boost',
	'FaceDataVisualizer': 'FF: Visualize Faces',
	'FaceMaskVisualizer': 'FF: Visualize Face Mask'
}
