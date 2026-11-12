"""Release the cached ONNX Runtime sessions when ComfyUI frees memory.

Every model (detector, swapper, occluder, parser, deep swapper, content filter) keeps its
InferenceSession in a module-level cache for the life of the process. On a GPU execution
provider (CUDA, ROCm, DirectML, ...) each session's memory arena stays allocated, and ComfyUI
does not know about it: neither "Free model and node cache" (POST /free) nor its own memory
manager can reclaim it, so the next large model load runs out of VRAM.

install_comfy_hooks() makes ComfyUI's unload_all_models() also drop these sessions, and lets
free_memory() drop them when unloading ComfyUI's own models still leaves too little free memory.
The sessions are created again on the next face swap.
"""
import gc
import logging
import sys
from typing import Callable, List


logger = logging.getLogger(__name__)
_HOOK_MARK = '_facefusion_releases_onnx_sessions'


def _caches() -> List[Callable[[], int]]:
	"""One clear function per session cache; each returns how many entries it dropped."""
	from .detection import detector
	from .models import deep_swapper, occluder, parser, swapper

	def clear_dict(module, name):
		def clear():
			cache = getattr(module, name)
			count = len(cache)
			cache.clear()
			return count
		return clear

	def clear_swapper():
		count = int(swapper._swapper_instance is not None)
		swapper._swapper_instance = None
		return count

	def clear_detector_anchors():
		# lru_cache on a method keeps a reference to every detector it was called on.
		detector.FaceDetector._create_anchors.cache_clear()
		return 0

	clears = [
		clear_dict(detector, '_detector_instances'),
		clear_detector_anchors,
		clear_swapper,
		clear_dict(occluder, '_occluder_instances'),
		clear_dict(parser, '_parser_instances'),
		clear_dict(deep_swapper, '_SWAPPER_CACHE'),
	]
	content_filter = next((m for n, m in sys.modules.items() if n.endswith('content_filter.content_filter')), None)
	if content_filter is not None and hasattr(content_filter, '_filter_instance'):
		def clear_filter():
			count = int(content_filter._filter_instance is not None)
			content_filter._filter_instance = None
			return count
		clears.append(clear_filter)
	return clears


def release_onnx_sessions() -> int:
	"""Drop every cached ONNX Runtime session. Returns the number of model instances released."""
	released = sum(clear() for clear in _caches())
	if released:
		gc.collect()
		logger.info('[FaceFusion] released %d cached ONNX model(s)', released)
	return released


def install_comfy_hooks() -> bool:
	"""Hook ComfyUI's model management. Returns False outside ComfyUI or if already hooked."""
	try:
		import comfy.model_management as mm
	except ImportError:
		return False
	if getattr(mm.unload_all_models, _HOOK_MARK, False):
		return False

	original_unload_all = mm.unload_all_models
	original_free_memory = mm.free_memory

	def unload_all_models(*args, **kwargs):
		result = original_unload_all(*args, **kwargs)
		if release_onnx_sessions():
			mm.soft_empty_cache(force=True)
		return result

	def free_memory(memory_required, device, *args, **kwargs):
		result = original_free_memory(memory_required, device, *args, **kwargs)
		try:
			short = device is not None and getattr(device, 'type', 'cpu') != 'cpu' \
				and mm.get_free_memory(device) < memory_required
		except Exception:
			short = False
		if short and release_onnx_sessions():
			mm.soft_empty_cache(force=True)
		return result

	for hook in (unload_all_models, free_memory):
		setattr(hook, _HOOK_MARK, True)
	mm.unload_all_models = unload_all_models
	mm.free_memory = free_memory
	return True
