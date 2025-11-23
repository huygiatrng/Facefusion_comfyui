"""
Shared constants for face swapping models.
"""
import numpy as np

# Model URLs from facefusion assets
MODEL_URLS = {
    # Swapper models - HyperSwap
    'hyperswap_1a_256': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.onnx',
    'hyperswap_1b_256': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1b_256.onnx',
    'hyperswap_1c_256': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1c_256.onnx',
    # Ghost models (ai-forever)
    'ghost_1_256': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_1_256.onnx',
    'ghost_2_256': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_2_256.onnx',
    'ghost_3_256': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_3_256.onnx',
    # HifiFace (GuijiAI)
    'hififace_unofficial_256': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/hififace_unofficial_256.onnx',
    # InsightFace models
    'inswapper_128': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx',
    'inswapper_128_fp16': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx',
    # Other swapper models
    'blendswap_256': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/blendswap_256.onnx',
    'simswap_256': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_256.onnx',
    'simswap_unofficial_512': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_unofficial_512.onnx',
    'uniface_256': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/uniface_256.onnx',
    # Embedding converter models (crossface)
    'crossface_ghost': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_ghost.onnx',
    'crossface_hififace': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_hififace.onnx',
    'crossface_simswap': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_simswap.onnx',
    # Face occluder models (xseg)
    'xseg_1': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.onnx',
    'xseg_2': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_2.onnx',
    'xseg_3': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.2.0/xseg_3.onnx',
    # Face parser models (bisenet)
    'bisenet_resnet_18': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/bisenet_resnet_18.onnx',
    'bisenet_resnet_34': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx',
}

# Model configurations with normalization parameters (matching facefusion-master)
MODEL_CONFIGS = {
    # Swapper models - HyperSwap (FaceFusion proprietary, best quality)
    'hyperswap_1a_256': {'size': (256, 256), 'template': 'arcface_128', 'type': 'hyperswap', 
                          'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
    'hyperswap_1b_256': {'size': (256, 256), 'template': 'arcface_128', 'type': 'hyperswap',
                          'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
    'hyperswap_1c_256': {'size': (256, 256), 'template': 'arcface_128', 'type': 'hyperswap',
                          'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
    # Ghost models (ai-forever, Apache-2.0 license) - need crossface_ghost converter
    'ghost_1_256': {'size': (256, 256), 'template': 'arcface_112_v1', 'type': 'ghost',
                     'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'converter': 'crossface_ghost'},
    'ghost_2_256': {'size': (256, 256), 'template': 'arcface_112_v1', 'type': 'ghost',
                     'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'converter': 'crossface_ghost'},
    'ghost_3_256': {'size': (256, 256), 'template': 'arcface_112_v1', 'type': 'ghost',
                     'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'converter': 'crossface_ghost'},
    # HifiFace (GuijiAI) - needs crossface_hififace converter
    'hififace_unofficial_256': {'size': (256, 256), 'template': 'mtcnn_512', 'type': 'hififace',
                                  'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'converter': 'crossface_hififace'},
    # InsightFace models (fast and widely used) - NO mean/std normalization (stays in [0,1])
    'inswapper_128': {'size': (128, 128), 'template': 'arcface_128', 'type': 'inswapper',
                       'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
    'inswapper_128_fp16': {'size': (128, 128), 'template': 'arcface_128', 'type': 'inswapper',
                            'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
    # Other swapper models
    'blendswap_256': {'size': (256, 256), 'template': 'ffhq_512', 'type': 'blendswap',
                       'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
    # SimSwap models - need crossface_simswap converter
    # simswap_256 uses ImageNet normalization
    'simswap_256': {'size': (256, 256), 'template': 'arcface_112_v1', 'type': 'simswap',
                     'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'converter': 'crossface_simswap'},
    # simswap_unofficial_512 uses NO normalization (mean=0, std=1)
    'simswap_unofficial_512': {'size': (512, 512), 'template': 'arcface_112_v1', 'type': 'simswap',
                                 'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0], 'converter': 'crossface_simswap'},
    'uniface_256': {'size': (256, 256), 'template': 'ffhq_512', 'type': 'uniface',
                     'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
    # Face occluder models (xseg) - output a mask of occluded regions
    'xseg_1': {'size': (256, 256), 'type': 'occluder'},
    'xseg_2': {'size': (256, 256), 'type': 'occluder'},
    'xseg_3': {'size': (256, 256), 'type': 'occluder'},
    # Face parser models (bisenet) - segment face into regions
    'bisenet_resnet_18': {'size': (512, 512), 'type': 'parser'},
    'bisenet_resnet_34': {'size': (512, 512), 'type': 'parser'},
}

# Warp templates for face alignment
WARP_TEMPLATES = {
    'arcface_112_v1': np.array([
        [0.35473214, 0.45658929],
        [0.64526786, 0.45658929],
        [0.50000000, 0.61154464],
        [0.37913393, 0.77687500],
        [0.62086607, 0.77687500]
    ], dtype=np.float32),
    'arcface_112_v2': np.array([
        [0.34191607, 0.46157411],
        [0.65653393, 0.45983393],
        [0.50022500, 0.64050536],
        [0.37097589, 0.82469196],
        [0.63151696, 0.82325089]
    ], dtype=np.float32),
    'arcface_128': np.array([
        [0.36167656, 0.40387734],
        [0.63696719, 0.40235469],
        [0.50019687, 0.56044219],
        [0.38710391, 0.72160547],
        [0.61507734, 0.72034453]
    ], dtype=np.float32),
    'ffhq_512': np.array([
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465]
    ], dtype=np.float32),
    'mtcnn_512': np.array([
        [0.34191607, 0.46157411],
        [0.65653393, 0.45983393],
        [0.50022500, 0.64050536],
        [0.37097589, 0.82469196],
        [0.63151696, 0.82325089]
    ], dtype=np.float32),
}






