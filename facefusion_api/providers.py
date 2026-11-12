"""onnxruntime execution provider selection shared by all ONNX models of this node pack."""
from typing import List

import onnxruntime as ort

# Preferred order. Only providers the installed onnxruntime build offers are used, so the CUDA
# build picks CUDA, onnxruntime-rocm picks ROCm, DirectML/CoreML/OpenVINO builds pick theirs and
# everything else runs on the CPU.
PREFERRED_PROVIDERS = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
)


def select_execution_providers() -> List[str]:
    available = set(ort.get_available_providers())
    return [provider for provider in PREFERRED_PROVIDERS if provider in available]
