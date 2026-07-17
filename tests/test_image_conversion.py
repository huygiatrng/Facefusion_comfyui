import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


utils = load_module("facefusion_test_utils", REPO_ROOT / "facefusion_api" / "utils.py")


class ImageConversionTests(unittest.TestCase):
    def test_tensor_to_cv2_clips_float_values_without_uint8_wraparound(self):
        image = torch.tensor([[[[-0.5, 0.5, 4.0]]]], dtype=torch.float32)

        result = utils.tensor_to_cv2(image)

        np.testing.assert_array_equal(
            result, np.array([[[255, 128, 0]]], dtype=np.uint8)
        )

    def test_unchanged_sdr_proxy_restores_original_hdr_tensor(self):
        reference = torch.tensor([[[[-0.25, 0.5, 4.0]]]], dtype=torch.float32)
        proxy = utils.tensor_to_cv2(reference)

        restored = utils.cv2_result_to_tensor(proxy, reference)

        torch.testing.assert_close(restored, reference, rtol=0, atol=0)

    def test_processed_delta_keeps_highlight_headroom(self):
        reference = torch.full((1, 1, 1, 3), 4.0, dtype=torch.float32)
        processed = utils.tensor_to_cv2(reference)
        processed[:] = 128

        restored = utils.cv2_result_to_tensor(processed, reference)

        self.assertGreater(float(restored.min()), 3.0)
        self.assertGreater(float(restored.max()), 1.0)


class DeepSwapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = types.ModuleType("facefusion_api")
        package.__path__ = [str(REPO_ROOT / "facefusion_api")]
        models_package = types.ModuleType("facefusion_api.models")
        models_package.__path__ = [str(REPO_ROOT / "facefusion_api" / "models")]
        detection = types.ModuleType("facefusion_api.detection")
        detection.detect_faces = lambda *args, **kwargs: []
        sys.modules["facefusion_api"] = package
        sys.modules["facefusion_api.models"] = models_package
        sys.modules["facefusion_api.detection"] = detection
        sys.modules["facefusion_api.utils"] = utils
        cls.deep_swapper = load_module(
            "facefusion_api.models.deep_swapper",
            REPO_ROOT / "facefusion_api" / "models" / "deep_swapper.py",
        )

    def test_model_discovery_and_safe_resolution(self):
        original_directory = self.deep_swapper.MODEL_DIRECTORY
        with tempfile.TemporaryDirectory() as temp_directory:
            model_directory = Path(temp_directory)
            (model_directory / "custom").mkdir()
            model_path = model_directory / "custom" / "identity.dfm"
            model_path.write_bytes(b"dfm")
            self.deep_swapper.MODEL_DIRECTORY = model_directory
            try:
                self.assertEqual(
                    self.deep_swapper.get_deep_swapper_model_names(),
                    ["custom/identity.dfm"],
                )
                self.assertEqual(
                    self.deep_swapper.resolve_deep_swapper_model_path(
                        "custom/identity.dfm"
                    ),
                    model_path.resolve(),
                )
                with self.assertRaises(ValueError):
                    self.deep_swapper.resolve_deep_swapper_model_path("../identity.dfm")
            finally:
                self.deep_swapper.MODEL_DIRECTORY = original_directory

    def test_dfm_forward_path_uses_morph_input_and_preserves_frame_shape(self):
        class FakeSession:
            def __init__(self):
                self.inputs = None

            def run(self, output_names, inputs):
                self.inputs = inputs
                face = np.full_like(inputs["in_face:0"], 0.5, dtype=np.float32)
                mask_shape = face.shape[:3] + (1,)
                mask = np.ones(mask_shape, dtype=np.float32)
                return mask, face, mask

        swapper = self.deep_swapper.DeepFaceLiveSwapper(Path("identity.dfm"))
        swapper.model_session = FakeSession()
        swapper.model_size = (32, 32)
        swapper.has_morph_input = True
        target = np.full((96, 96, 3), 64, dtype=np.uint8)
        landmarks = self.deep_swapper.DFL_WHOLE_FACE_TEMPLATE * np.array(
            [96, 96], dtype=np.float32
        )
        target_face = {"landmarks": landmarks}

        result = swapper.swap_face(target_face, target, morph=25)

        self.assertEqual(result.shape, target.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertIn("morph_value:0", swapper.model_session.inputs)
        np.testing.assert_allclose(
            swapper.model_session.inputs["morph_value:0"], [0.25]
        )


if __name__ == "__main__":
    unittest.main()
