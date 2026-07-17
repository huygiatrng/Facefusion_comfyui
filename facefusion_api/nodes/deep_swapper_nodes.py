"""ComfyUI nodes for DeepFaceLive DFM models."""

from typing import Tuple

import torch
from comfy.comfy_types import IO
from torch import Tensor

from ..models.deep_swapper import get_deep_swapper_model_names, swap_deep_faces_local
from ..types import InputTypes
from ..utils import cv2_result_to_tensor, cv2_to_tensor, tensor_to_cv2
from .content_filter_utils import CONTENT_FILTER_AVAILABLE, analyse_frame, blur_frame


class DeepSwapFaceImage:
    """Swap detected faces with an identity-specific DeepFaceLive DFM model."""

    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "target_image": (IO.IMAGE,),
                "dfm_model": (get_deep_swapper_model_names(),),
                "morph": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "face_detector_model": (
                    ["scrfd", "retinaface", "yolo_face", "yunet", "many"],
                    {"default": "scrfd"},
                ),
                "face_selector_mode": (["one", "many"], {"default": "one"}),
                "face_position": ("INT", {"default": 0, "min": 0, "max": 100}),
                "sort_order": (
                    [
                        "large-small",
                        "small-large",
                        "left-right",
                        "right-left",
                        "top-bottom",
                        "bottom-top",
                        "best-worst",
                        "worst-best",
                    ],
                    {"default": "large-small"},
                ),
                "score_threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "face_mask_blur": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "face_mask_padding": (
                    "STRING",
                    {"default": "0,0,0,0", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "process"
    CATEGORY = "FaceFusion"

    @staticmethod
    def process(
        target_image: Tensor,
        dfm_model: str,
        morph: int,
        face_detector_model: str,
        face_selector_mode: str,
        face_position: int,
        sort_order: str,
        score_threshold: float,
        face_mask_blur: float,
        face_mask_padding: str,
    ) -> Tuple[Tensor]:
        try:
            padding = tuple(
                int(value.strip()) for value in face_mask_padding.split(",")
            )
        except ValueError as error:
            raise ValueError(
                "face_mask_padding must contain four comma-separated integers."
            ) from error
        if len(padding) != 4:
            raise ValueError(
                "face_mask_padding must contain four comma-separated integers."
            )

        targets = target_image if target_image.dim() == 4 else target_image.unsqueeze(0)
        output_images = []

        for index in range(targets.shape[0]):
            single_target = targets[index : index + 1]
            target_cv2 = tensor_to_cv2(single_target)

            if CONTENT_FILTER_AVAILABLE and analyse_frame(target_cv2):
                output_images.append(cv2_to_tensor(blur_frame(target_cv2)))
                continue

            result_cv2 = swap_deep_faces_local(
                target_image=target_cv2,
                model_name=dfm_model,
                morph=morph,
                face_selector_mode=face_selector_mode,
                face_position=face_position,
                sort_order=sort_order,
                score_threshold=score_threshold,
                face_detector_model=face_detector_model,
                face_mask_blur=face_mask_blur,
                face_mask_padding=padding,
            )
            output_images.append(cv2_result_to_tensor(result_cv2, single_target))

        return (torch.cat(output_images, dim=0),)
