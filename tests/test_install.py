import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_install():
    spec = importlib.util.spec_from_file_location("facefusion_install", REPO_ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["facefusion_install"] = module
    spec.loader.exec_module(module)
    return module


class OnnxRuntimePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.install = load_install()

    def test_accelerated_builds_are_kept(self):
        for build in ("onnxruntime-rocm", "onnxruntime-directml", "onnxruntime-openvino",
                      "onnxruntime-silicon", "onnxruntime-qnn", "onnxruntime-gpu"):
            for torch_uses_cuda in (True, False):
                with self.subTest(build=build, torch_uses_cuda=torch_uses_cuda):
                    self.assertEqual(self.install.onnxruntime_plan({build}, torch_uses_cuda), ([], []))

    def test_cpu_build_is_replaced_by_the_cuda_build_on_cuda_torch(self):
        self.assertEqual(self.install.onnxruntime_plan({"onnxruntime"}, True),
                         (["onnxruntime"], ["onnxruntime-gpu"]))

    def test_missing_onnxruntime_on_cuda_torch_gets_the_cuda_build(self):
        self.assertEqual(self.install.onnxruntime_plan(set(), True), ([], ["onnxruntime-gpu"]))

    def test_without_cuda_torch_the_cuda_build_is_never_installed(self):
        self.assertEqual(self.install.onnxruntime_plan({"onnxruntime"}, False), ([], []))
        self.assertEqual(self.install.onnxruntime_plan(set(), False), ([], ["onnxruntime"]))


class InstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.install = load_install()

    def _run(self, installed, torch_uses_cuda, env=None):
        calls = []
        with mock.patch.object(self.install, "_installed", side_effect=lambda name: name in installed), \
                mock.patch.object(self.install, "_torch_uses_cuda", return_value=torch_uses_cuda), \
                mock.patch.object(self.install, "_pip", side_effect=lambda *args: calls.append(args)), \
                mock.patch.object(self.install.os.path, "exists", return_value=False), \
                mock.patch("builtins.open", mock.mock_open()), \
                mock.patch.dict(self.install.os.environ, env or {}, clear=False):
            self.install.install()
        return calls

    def test_rocm_onnxruntime_is_not_touched(self):
        calls = self._run({"onnxruntime-rocm", "onnx", "httpx", "httpx-retries"}, torch_uses_cuda=False)
        self.assertEqual(calls, [])

    def test_only_missing_helpers_are_installed_next_to_rocm_onnxruntime(self):
        calls = self._run({"onnxruntime-rocm"}, torch_uses_cuda=False)
        self.assertEqual(calls, [("install", "httpx==0.28.1"), ("install", "httpx-retries==0.4.0"),
                                 ("install", "onnx")])

    def test_cuda_machine_with_cpu_onnxruntime_switches_to_the_cuda_build(self):
        calls = self._run({"onnxruntime", "onnx", "httpx", "httpx-retries"}, torch_uses_cuda=True)
        self.assertEqual(calls, [("uninstall", "-y", "onnxruntime"), ("install", "onnxruntime-gpu")])

    def test_opt_out_skips_everything(self):
        calls = self._run(set(), torch_uses_cuda=True, env={"FACEFUSION_SKIP_ONNX_INSTALL": "1"})
        self.assertEqual(calls, [])

    def test_pip_runs_from_the_current_interpreter(self):
        with mock.patch.object(self.install.subprocess, "run") as run:
            self.install._pip("install", "onnx")
        self.assertEqual(run.call_args.args[0][:3], [sys.executable, "-m", "pip"])


if __name__ == "__main__":
    unittest.main()
