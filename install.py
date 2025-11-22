import subprocess
from shutil import which


def install() -> None:
	subprocess.run([ which('pip'), 'install', 'httpx==0.28.1', '-q' ])
	subprocess.run([ which('pip'), 'install', 'httpx-retries==0.4.0', '-q' ])
