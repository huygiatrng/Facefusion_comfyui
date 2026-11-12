import os
import subprocess
import sys
from importlib import metadata


# onnxruntime builds that bring their own GPU/NPU execution provider. If one of them is
# installed, it was chosen for this machine and must not be replaced by the CUDA build.
ACCELERATED_ONNXRUNTIME = (
	'onnxruntime-gpu',
	'onnxruntime-rocm',
	'onnxruntime-directml',
	'onnxruntime-openvino',
	'onnxruntime-silicon',
	'onnxruntime-qnn',
)


def _installed(distribution: str) -> bool:
	try:
		metadata.version(distribution)
		return True
	except metadata.PackageNotFoundError:
		return False


def _torch_uses_cuda() -> bool:
	"""True only for an NVIDIA CUDA build of torch with a visible GPU (not ROCm, not CPU-only)."""
	try:
		import torch
		return torch.version.cuda is not None and getattr(torch.version, 'hip', None) is None and torch.cuda.is_available()
	except Exception:
		return False


def onnxruntime_plan(installed, torch_uses_cuda: bool):
	"""Return (uninstall, install) package lists for onnxruntime.

	``installed`` is the set of installed distribution names. An accelerated build is kept as is.
	The CUDA build is only installed when torch itself runs on CUDA; elsewhere a missing onnxruntime
	becomes the CPU build, which works everywhere.
	"""
	if any(name in installed for name in ACCELERATED_ONNXRUNTIME):
		return [], []
	if torch_uses_cuda:
		return (['onnxruntime'] if 'onnxruntime' in installed else []), ['onnxruntime-gpu']
	return [], ([] if 'onnxruntime' in installed else ['onnxruntime'])


def _pip(*args: str) -> None:
	# The pip of the Python running ComfyUI, not whichever pip comes first on PATH.
	subprocess.run([sys.executable, '-m', 'pip', *args, '-q'], check=False)


def install() -> None:
	"""Install dependencies only if needed."""
	if os.environ.get('FACEFUSION_SKIP_ONNX_INSTALL') == '1':
		return
	marker_file = os.path.join(os.path.dirname(__file__), '.install_complete')

	try:
		candidates = ('onnxruntime', 'onnx', 'httpx', 'httpx-retries') + ACCELERATED_ONNXRUNTIME
		installed = {name for name in candidates if _installed(name)}
		uninstall, onnx_install = onnxruntime_plan(installed, _torch_uses_cuda())
		missing = [name for name in ('httpx', 'httpx-retries', 'onnx') if name not in installed]

		if not (uninstall or onnx_install or missing) and os.path.exists(marker_file):
			# Already installed correctly, skip
			return

		print("[Facefusion_comfyui] Setting up ONNX dependencies...")
		if uninstall:
			_pip('uninstall', '-y', *uninstall)
		pinned = {'httpx': 'httpx==0.28.1', 'httpx-retries': 'httpx-retries==0.4.0'}
		for name in missing + onnx_install:
			_pip('install', pinned.get(name, name))

		# Create marker file
		with open(marker_file, 'w') as f:
			f.write('Installation complete')

		print("[Facefusion_comfyui] Dependencies installed successfully!")

	except Exception as e:
		print(f"[Facefusion_comfyui] Warning during installation: {e}")
