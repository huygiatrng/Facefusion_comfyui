"""Reproduce: the face swap's ONNX Runtime sessions keep their GPU memory after ComfyUI frees memory.

Runs FaceDetectorNode + FaceSwapApplier once on a GPU execution provider, then does what
ComfyUI's POST /free does (unload_all_models + soft_empty_cache) and reports this process's
VRAM before, after the swap and after the free. Without the fix the swap's memory stays.
A second step asks ComfyUI's free_memory() for (almost) all of the GPU, as a large model
load would, and reports whether the memory then becomes available.

Run with ComfyUI's Python from a directory that is not the repository:
  python tests/manual/repro_vram_release.py --comfyui ~/ComfyUI --node . --source face.jpg --target photo.jpg
Needs a GPU onnxruntime build (onnxruntime-gpu, onnxruntime-rocm, ...). VRAM is read from the
amdgpu/i915/xe DRM fdinfo of this process (Linux); on NVIDIA it falls back to nvidia-smi.
"""
import argparse
import importlib.util
import os
import subprocess
import sys


def process_vram_mib():
	seen, total = set(), 0
	fd_dir = f'/proc/{os.getpid()}/fdinfo'
	for fd in os.listdir(fd_dir):
		try:
			info = open(f'{fd_dir}/{fd}').read()
		except OSError:
			continue
		fields = dict(line.split(':', 1) for line in info.splitlines() if ':' in line)
		client = fields.get('drm-client-id', '').strip()
		vram = fields.get('drm-memory-vram', '').strip()
		if client and vram and client not in seen:
			seen.add(client)
			total += int(vram.split()[0]) // 1024
	if seen:
		return total
	out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
		capture_output=True, text=True).stdout
	return sum(int(m) for p, m in (line.split(', ') for line in out.splitlines()) if int(p) == os.getpid())


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--comfyui', required=True)
	ap.add_argument('--node', required=True, help='Facefusion_comfyui checkout')
	ap.add_argument('--source', required=True, help='image with the face to use')
	ap.add_argument('--target', required=True, help='image with the face to replace')
	a = ap.parse_args()

	sys.argv = [sys.argv[0]]
	sys.path.insert(0, a.comfyui)
	os.environ.setdefault('FACEFUSION_SKIP_ONNX_INSTALL', '1')
	import numpy as np
	import torch
	from PIL import Image
	import comfy.model_management as mm
	import folder_paths  # noqa: F401  (models/ paths for the node)

	spec = importlib.util.spec_from_file_location('facefusion_comfyui', os.path.join(a.node, '__init__.py'),
		submodule_search_locations=[a.node])
	node = importlib.util.module_from_spec(spec)
	sys.modules['facefusion_comfyui'] = node
	spec.loader.exec_module(node)
	nodes = node.NODE_CLASS_MAPPINGS
	hooked = getattr(mm.unload_all_models, '_facefusion_releases_onnx_sessions', False)

	def load(path):
		return torch.from_numpy(np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0)[None]

	import onnxruntime as ort
	print(f'onnxruntime {ort.__version__} {ort.get_available_providers()}  hooks installed: {hooked}')
	torch.zeros(1, device=mm.get_torch_device())  # create the torch context first, like ComfyUI does
	v0 = process_vram_mib()
	face_data = nodes['FaceDetectorNode']().detect(load(a.target), 'scrfd', 'one', 0, 'large-small', 0.3)[0]
	out = nodes['FaceSwapApplier']().apply(load(a.source), face_data, '-1', 'hyperswap_1c_256', '1024x1024',
		'xseg_1', 'bisenet_resnet_34', 0.3, 0)
	assert out[0].shape == load(a.target).shape, 'face swap did not return an image'
	v1 = process_vram_mib()
	mm.unload_all_models()
	mm.soft_empty_cache(force=True)
	v2 = process_vram_mib()
	print(f'process VRAM: before swap {v0} MiB, after swap {v1} MiB, after ComfyUI free {v2} MiB')

	# A big model load asks free_memory() for most of the GPU.
	dev = mm.get_torch_device()
	nodes['FaceSwapApplier']().apply(load(a.source), face_data, '-1', 'hyperswap_1c_256', '1024x1024',
		'xseg_1', 'bisenet_resnet_34', 0.3, 0)
	before = mm.get_free_memory(dev)
	need = mm.get_total_memory(dev) - 512 * 1024 ** 2
	mm.free_memory(need, dev)
	after = mm.get_free_memory(dev)
	print(f'free_memory({need >> 20} MiB): device free {before >> 20} -> {after >> 20} MiB, '
		f'process VRAM {process_vram_mib()} MiB')
	released = v1 - v2
	print('RESULT:', 'released' if released > 0.5 * (v1 - v0) else 'NOT released', f'({released} of {v1 - v0} MiB)')


if __name__ == '__main__':
	main()
