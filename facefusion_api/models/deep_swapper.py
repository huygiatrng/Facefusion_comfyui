"""DeepFaceLive DFM support based on FaceFusion's deep_swapper processor."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from ..detection import detect_faces
from ..utils import Face, VisionFrame


MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models" / "deep_swapper"
NO_DFM_MODELS = "[no .dfm models found]"
DFL_WHOLE_FACE_TEMPLATE = np.array(
    [
        [0.35342266, 0.39285716],
        [0.62797622, 0.39285716],
        [0.48660713, 0.54017860],
        [0.38839287, 0.68750011],
        [0.59821427, 0.68750011],
    ],
    dtype=np.float32,
)


def get_deep_swapper_model_names() -> List[str]:
    """Return DFM model paths relative to the custom node model directory."""
    if not MODEL_DIRECTORY.is_dir():
        return [NO_DFM_MODELS]

    model_names = [
        path.relative_to(MODEL_DIRECTORY).as_posix()
        for path in MODEL_DIRECTORY.rglob("*")
        if path.is_file() and path.suffix.lower() == ".dfm"
    ]
    return sorted(model_names, key=str.casefold) or [NO_DFM_MODELS]


def resolve_deep_swapper_model_path(model_name: str) -> Path:
    """Resolve a dropdown value without allowing paths outside the model folder."""
    if not model_name or model_name == NO_DFM_MODELS:
        raise FileNotFoundError(
            f"No DFM model is installed. Place a .dfm file in {MODEL_DIRECTORY} and restart ComfyUI."
        )

    model_directory = MODEL_DIRECTORY.resolve()
    model_path = (model_directory / model_name.replace("/", os.sep)).resolve()

    if not model_path.is_relative_to(model_directory):
        raise ValueError(
            "DFM model path must stay inside the FaceFusion deep_swapper model directory."
        )
    if model_path.suffix.lower() != ".dfm" or not model_path.is_file():
        raise FileNotFoundError(f"DFM model not found: {model_path}")
    return model_path


def _select_execution_providers() -> List[str]:
    available_providers = set(ort.get_available_providers())
    preferred_providers = [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [
        provider for provider in preferred_providers if provider in available_providers
    ]


class DeepFaceLiveSwapper:
    """Run a DeepFaceLive ``.dfm`` identity model on detected target faces."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model_session = None
        self.model_size: Tuple[int, int] = (0, 0)
        self.has_morph_input = False

    def initialize(self) -> None:
        providers = _select_execution_providers()
        self.model_session = ort.InferenceSession(
            str(self.model_path), providers=providers
        )
        model_inputs = {
            model_input.name: model_input
            for model_input in self.model_session.get_inputs()
        }
        face_input = model_inputs.get("in_face:0")

        if face_input is None or len(face_input.shape) < 4:
            raise ValueError(
                f"{self.model_path.name} is not a supported DeepFaceLive DFM model."
            )

        height, width = face_input.shape[1:3]
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(
                f"{self.model_path.name} has a dynamic or invalid face input size."
            )

        self.model_size = (height, width)
        self.has_morph_input = "morph_value:0" in model_inputs

    def swap_face(
        self,
        target_face: Face,
        target_image: VisionFrame,
        morph: int = 100,
        face_mask_blur: float = 0.3,
        face_mask_padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> VisionFrame:
        if self.model_session is None:
            self.initialize()

        crop_frame, affine_matrix = self._warp_face(
            target_image, target_face["landmarks"]
        )
        crop_frame_raw = crop_frame.copy()
        box_mask = self._create_box_mask(
            crop_frame.shape[:2], face_mask_blur, face_mask_padding
        )
        crop_frame = self._prepare_crop_frame(crop_frame)
        crop_frame, source_mask, target_mask = self._forward(crop_frame, morph)
        crop_frame = self._normalize_crop_frame(crop_frame)
        crop_frame = self._conditional_match_frame_color(crop_frame_raw, crop_frame)
        model_mask = self._prepare_crop_mask(source_mask, target_mask)
        crop_mask = np.minimum(box_mask, model_mask).clip(0, 1)
        return self._paste_back(target_image, crop_frame, crop_mask, affine_matrix)

    def _warp_face(
        self, image: VisionFrame, landmarks: NDArray
    ) -> Tuple[VisionFrame, NDArray]:
        height, width = self.model_size
        template = DFL_WHOLE_FACE_TEMPLATE * np.array([width, height], dtype=np.float32)
        affine_matrix = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32),
            template,
            method=cv2.RANSAC,
            ransacReprojThreshold=100,
        )[0]
        if affine_matrix is None:
            raise ValueError("Unable to align the selected face for the DFM model.")
        crop_frame = cv2.warpAffine(
            image,
            affine_matrix,
            (width, height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_AREA,
        )
        return crop_frame, affine_matrix

    @staticmethod
    def _prepare_crop_frame(crop_frame: VisionFrame) -> VisionFrame:
        crop_frame = cv2.addWeighted(
            crop_frame,
            1.75,
            cv2.GaussianBlur(crop_frame, (0, 0), 2),
            -0.75,
            0,
        )
        return np.expand_dims(crop_frame / 255.0, axis=0).astype(np.float32)

    def _forward(
        self, crop_frame: VisionFrame, morph: int
    ) -> Tuple[VisionFrame, NDArray, NDArray]:
        model_inputs = {"in_face:0": crop_frame}
        if self.has_morph_input:
            model_inputs["morph_value:0"] = np.array(
                [np.clip(morph, 0, 100) / 100.0], dtype=np.float32
            )

        outputs = self.model_session.run(None, model_inputs)
        if len(outputs) < 3:
            raise ValueError(
                f"{self.model_path.name} returned an unsupported DFM output layout."
            )

        target_mask, output_frame, source_mask = outputs[:3]
        return output_frame[0], source_mask[0], target_mask[0]

    @staticmethod
    def _normalize_crop_frame(crop_frame: VisionFrame) -> VisionFrame:
        if (
            crop_frame.ndim == 3
            and crop_frame.shape[0] in (1, 3, 4)
            and crop_frame.shape[-1] not in (1, 3, 4)
        ):
            crop_frame = crop_frame.transpose(1, 2, 0)
        return (crop_frame * 255.0).clip(0, 255).astype(np.uint8)

    def _prepare_crop_mask(self, source_mask: NDArray, target_mask: NDArray) -> NDArray:
        height, width = self.model_size
        crop_mask = np.minimum(source_mask, target_mask).squeeze()
        if crop_mask.size != height * width:
            raise ValueError(
                f"{self.model_path.name} returned an unsupported DFM mask layout."
            )
        crop_mask = crop_mask.reshape((height, width)).clip(0, 1).astype(np.float32)
        crop_mask = cv2.erode(
            crop_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=2,
        )
        return cv2.GaussianBlur(crop_mask, (0, 0), 6.25)

    @staticmethod
    def _create_box_mask(
        size: Tuple[int, int],
        face_mask_blur: float,
        padding: Tuple[int, int, int, int],
    ) -> NDArray:
        height, width = size
        blur_amount = int(width * 0.5 * face_mask_blur)
        blur_area = max(blur_amount // 2, 1)
        mask = np.ones((height, width), dtype=np.float32)
        top, right, bottom, left = padding
        mask[: max(blur_area, int(height * top / 100)), :] = 0
        mask[-max(blur_area, int(height * bottom / 100)) :, :] = 0
        mask[:, : max(blur_area, int(width * left / 100))] = 0
        mask[:, -max(blur_area, int(width * right / 100)) :] = 0
        if blur_amount > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
        return mask

    @staticmethod
    def _conditional_match_frame_color(
        source_frame: VisionFrame, target_frame: VisionFrame
    ) -> VisionFrame:
        histogram_source = cv2.calcHist(
            [cv2.cvtColor(source_frame, cv2.COLOR_BGR2HSV)],
            [0, 1],
            None,
            [50, 60],
            [0, 180, 0, 256],
        )
        histogram_target = cv2.calcHist(
            [cv2.cvtColor(target_frame, cv2.COLOR_BGR2HSV)],
            [0, 1],
            None,
            [50, 60],
            [0, 180, 0, 256],
        )
        histogram_factor = float(
            np.interp(
                cv2.compareHist(histogram_source, histogram_target, cv2.HISTCMP_CORREL),
                [-1, 1],
                [0, 1],
            )
        )

        matched_frame = target_frame
        for color_size in np.linspace(16, target_frame.shape[0], 3, endpoint=False):
            source_resize = cv2.resize(
                source_frame,
                (int(color_size), int(color_size)),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            target_resize = cv2.resize(
                matched_frame,
                (int(color_size), int(color_size)),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            color_difference = cv2.resize(
                source_resize - target_resize,
                target_frame.shape[:2][::-1],
                interpolation=cv2.INTER_CUBIC,
            )
            matched_frame = (
                (matched_frame + color_difference).clip(0, 255).astype(np.uint8)
            )

        return cv2.addWeighted(
            target_frame, 1 - histogram_factor, matched_frame, histogram_factor, 0
        )

    @staticmethod
    def _paste_back(
        target_image: VisionFrame,
        crop_frame: VisionFrame,
        mask: NDArray,
        affine_matrix: NDArray,
    ) -> VisionFrame:
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        crop_height, crop_width = crop_frame.shape[:2]
        target_height, target_width = target_image.shape[:2]
        corners = np.array(
            [[0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height]],
            dtype=np.float32,
        )
        corners = cv2.transform(corners.reshape(1, -1, 2), inverse_matrix).reshape(
            -1, 2
        )
        x_min = max(0, int(np.floor(corners[:, 0].min())))
        y_min = max(0, int(np.floor(corners[:, 1].min())))
        x_max = min(target_width, int(np.ceil(corners[:, 0].max())))
        y_max = min(target_height, int(np.ceil(corners[:, 1].max())))
        paste_width = x_max - x_min
        paste_height = y_max - y_min

        if paste_width <= 0 or paste_height <= 0:
            return target_image

        paste_matrix = inverse_matrix.copy()
        paste_matrix[0, 2] -= x_min
        paste_matrix[1, 2] -= y_min
        warped_crop = cv2.warpAffine(
            crop_frame, paste_matrix, (paste_width, paste_height)
        )
        warped_mask = cv2.warpAffine(mask, paste_matrix, (paste_width, paste_height))[
            ..., None
        ]
        result = target_image.copy()
        paste_region = result[y_min:y_max, x_min:x_max]
        result[y_min:y_max, x_min:x_max] = (
            paste_region * (1 - warped_mask) + warped_crop * warped_mask
        ).astype(np.uint8)
        return result


_SWAPPER_CACHE: Dict[Tuple[Path, int], DeepFaceLiveSwapper] = {}


def get_deep_swapper(model_name: str) -> DeepFaceLiveSwapper:
    model_path = resolve_deep_swapper_model_path(model_name)
    cache_key = (model_path, model_path.stat().st_mtime_ns)

    for old_key in list(_SWAPPER_CACHE):
        if old_key[0] == model_path and old_key != cache_key:
            del _SWAPPER_CACHE[old_key]

    if cache_key not in _SWAPPER_CACHE:
        _SWAPPER_CACHE[cache_key] = DeepFaceLiveSwapper(model_path)
    return _SWAPPER_CACHE[cache_key]


def swap_deep_faces_local(
    target_image: VisionFrame,
    model_name: str,
    morph: int = 100,
    face_selector_mode: str = "one",
    face_position: int = 0,
    sort_order: str = "large-small",
    score_threshold: float = 0.3,
    face_detector_model: str = "scrfd",
    face_mask_blur: float = 0.3,
    face_mask_padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> VisionFrame:
    target_faces = detect_faces(
        target_image, score_threshold, sort_order, face_detector_model
    )
    if not target_faces:
        return target_image

    if face_selector_mode == "many":
        selected_faces = target_faces
    else:
        selected_faces = [target_faces[min(face_position, len(target_faces) - 1)]]

    swapper = get_deep_swapper(model_name)
    result = target_image.copy()
    for target_face in selected_faces:
        result = swapper.swap_face(
            target_face,
            result,
            morph=morph,
            face_mask_blur=face_mask_blur,
            face_mask_padding=face_mask_padding,
        )
    return result
