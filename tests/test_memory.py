import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'facefusion_api_memory_test'


def load_memory():
	"""Import facefusion_api.memory and the cache modules without running the package __init__
	(which imports ComfyUI and registers the nodes)."""
	# A private package name keeps this independent of stubs other tests put in sys.modules.
	if PACKAGE not in sys.modules:
		package = types.ModuleType(PACKAGE)
		package.__path__ = [str(REPO_ROOT / 'facefusion_api')]
		sys.modules[PACKAGE] = package
	return importlib.import_module(PACKAGE + '.memory')


def submodule(name):
	return importlib.import_module(f'{PACKAGE}.{name}')


class FakeModelManagement(types.ModuleType):
	def __init__(self, free_after=0):
		super().__init__('comfy.model_management')
		self.calls = []
		self.free_after = free_after
		self.unload_all_models = lambda: self.calls.append('unload_all')
		self.free_memory = lambda memory_required, device, *a, **k: self.calls.append(('free', memory_required))
		self.get_free_memory = lambda device=None: self.free_after
		self.soft_empty_cache = lambda force=False: self.calls.append('empty_cache')


class ReleaseTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.memory = load_memory()

	def test_every_session_cache_is_cleared(self):
		detector = submodule('detection.detector')
		deep_swapper, occluder, parser, swapper = (submodule('models.' + n) for n in
			('deep_swapper', 'occluder', 'parser', 'swapper'))
		detector._detector_instances['scrfd_2.5g'] = object()
		occluder._occluder_instances['xseg_1'] = object()
		parser._parser_instances['bisenet_resnet_34'] = object()
		deep_swapper._SWAPPER_CACHE[('m', 1)] = object()
		swapper._swapper_instance = object()
		with mock.patch.object(detector.FaceDetector._create_anchors, 'cache_clear') as anchors:
			self.assertEqual(self.memory.release_onnx_sessions(), 5)
		anchors.assert_called_once_with()
		self.assertEqual((detector._detector_instances, occluder._occluder_instances, parser._parser_instances,
			deep_swapper._SWAPPER_CACHE, swapper._swapper_instance), ({}, {}, {}, {}, None))

	def test_nothing_cached_releases_nothing(self):
		self.assertEqual(self.memory.release_onnx_sessions(), 0)


class HookTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.memory = load_memory()

	def _install(self, free_after):
		mm = FakeModelManagement(free_after)
		comfy = types.ModuleType('comfy')
		comfy.model_management = mm
		with mock.patch.dict(sys.modules, {'comfy': comfy, 'comfy.model_management': mm}):
			installed = self.memory.install_comfy_hooks()
			again = self.memory.install_comfy_hooks()
		return mm, installed, again

	def test_hooks_install_once(self):
		_, installed, again = self._install(0)
		self.assertTrue(installed)
		self.assertFalse(again)

	def test_without_comfyui_nothing_is_hooked(self):
		with mock.patch.dict(sys.modules, {'comfy': None, 'comfy.model_management': None}):
			self.assertFalse(self.memory.install_comfy_hooks())

	def test_unload_all_models_also_releases_sessions(self):
		mm, _, _ = self._install(0)
		with mock.patch.object(self.memory, 'release_onnx_sessions', return_value=3) as release:
			mm.unload_all_models()
		release.assert_called_once_with()
		self.assertEqual(mm.calls, ['unload_all', 'empty_cache'])

	def test_free_memory_releases_only_when_still_short(self):
		gpu = types.SimpleNamespace(type='cuda')
		mm, _, _ = self._install(free_after=10)
		with mock.patch.object(self.memory, 'release_onnx_sessions', return_value=1) as release:
			mm.free_memory(5, gpu)
			release.assert_not_called()
			mm.free_memory(50, gpu)
			release.assert_called_once_with()
			mm.free_memory(50, types.SimpleNamespace(type='cpu'))
			release.assert_called_once_with()
		self.assertEqual(mm.calls, [('free', 5), ('free', 50), 'empty_cache', ('free', 50)])


if __name__ == '__main__':
	unittest.main()
