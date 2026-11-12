import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_providers(available):
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.get_available_providers = lambda: list(available)
    sys.modules["onnxruntime"] = fake_ort
    spec = importlib.util.spec_from_file_location("facefusion_providers", REPO_ROOT / "facefusion_api" / "providers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExecutionProviderTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("onnxruntime", None)

    def test_rocm_build_uses_rocm_not_migraphx(self):
        providers = load_providers(["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(providers.select_execution_providers(), ["ROCMExecutionProvider", "CPUExecutionProvider"])

    def test_cuda_build_keeps_cuda_first(self):
        providers = load_providers(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(providers.select_execution_providers(), ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_directml_and_cpu_only_builds(self):
        self.assertEqual(load_providers(["DmlExecutionProvider", "CPUExecutionProvider"]).select_execution_providers(),
                         ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(load_providers(["CPUExecutionProvider"]).select_execution_providers(),
                         ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
