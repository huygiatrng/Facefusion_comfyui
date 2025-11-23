import subprocess
from shutil import which


def install() -> None:
	# Uninstall conflicting ONNX packages first
	print("[Facefusion_comfyui] Uninstalling conflicting ONNX packages...")
	subprocess.run([ which('pip'), 'uninstall', 'onnx', 'onnxruntime', 'onnxruntime-gpu', '-y', '-q' ], stderr=subprocess.DEVNULL)
	
	# Install required packages
	print("[Facefusion_comfyui] Installing required packages...")
	subprocess.run([ which('pip'), 'install', 'httpx==0.28.1', '-q' ])
	subprocess.run([ which('pip'), 'install', 'httpx-retries==0.4.0', '-q' ])
	subprocess.run([ which('pip'), 'install', 'onnx', '-q' ])
	subprocess.run([ which('pip'), 'install', 'onnxruntime-gpu', '-q' ])
	print("[Facefusion_comfyui] Installation complete!")
