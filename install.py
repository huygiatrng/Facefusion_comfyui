import subprocess
from shutil import which
import os


def install() -> None:
	"""Install dependencies only if needed."""
	# Check if we've already done a successful install
	marker_file = os.path.join(os.path.dirname(__file__), '.install_complete')
	
	try:
		# Check if onnxruntime-gpu is already installed
		result = subprocess.run(
			[which('pip'), 'show', 'onnxruntime-gpu'],
			capture_output=True,
			text=True
		)
		
		if result.returncode == 0 and os.path.exists(marker_file):
			# Already installed correctly, skip
			return
		
		print("[Facefusion_comfyui] Setting up ONNX dependencies...")
		
		# Uninstall conflicting ONNX packages
		subprocess.run(
			[which('pip'), 'uninstall', 'onnx', 'onnxruntime', 'onnxruntime-gpu', '-y', '-q'],
			stderr=subprocess.DEVNULL
		)
		
		# Install required packages
		subprocess.run([which('pip'), 'install', 'httpx==0.28.1', '-q'])
		subprocess.run([which('pip'), 'install', 'httpx-retries==0.4.0', '-q'])
		subprocess.run([which('pip'), 'install', 'onnx', '-q'])
		subprocess.run([which('pip'), 'install', 'onnxruntime-gpu', '-q'])
		
		# Create marker file
		with open(marker_file, 'w') as f:
			f.write('Installation complete')
		
		print("[Facefusion_comfyui] Dependencies installed successfully!")
		
	except Exception as e:
		print(f"[Facefusion_comfyui] Warning during installation: {e}")
