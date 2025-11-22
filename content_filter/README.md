# Content Filter Module

This module implements NSFW content detection for FaceFusion ComfyUI nodes.

## How It Works

### Multi-Model Detection
- Uses 3 different ONNX models (`nsfw_1`, `nsfw_2`, `nsfw_3`) for NSFW detection
- Majority voting: Content is flagged as NSFW only if **at least 2 out of 3** models agree
- This reduces false positives while maintaining strong detection

### Hash Validation
- All model files are validated using CRC32 hashes
- The content filter Python module validates its own integrity on startup
- This prevents tampering with the detection logic

### Behavior
- **NSFW Detected**: Returns a heavily blurred version of the target image
- **Safe Content**: Processes normally
- Detection runs automatically on both source and target images in all swapper nodes

## Models

### nsfw_1
- **Size**: 640x640
- **License**: Apache-2.0
- **Vendor**: EraX

### nsfw_2
- **Size**: 384x384
- **License**: Apache-2.0
- **Vendor**: Marqo

### nsfw_3
- **Size**: 448x448
- **License**: MIT
- **Vendor**: Freepik

## Files

- `content_filter.py`: Main NSFW detection logic
- `hash_helper.py`: Hash validation utilities
- `__init__.py`: Module exports

## Integration

The content filter is automatically integrated into all face swapping nodes:
- `SwapFaceImage`
- `SwapFaceVideo`
- `AdvancedSwapFaceImage`
- `AdvancedSwapFaceVideo`
- `FaceSwapApplier`

No additional configuration is required.

## Model Storage

Models are automatically downloaded to:
```
custom_nodes/Facefusion_comfyui/models/content_filter/
```

On first run, the filter will download:
- `nsfw_1.onnx` + `nsfw_1.hash`
- `nsfw_2.onnx` + `nsfw_2.hash`
- `nsfw_3.onnx` + `nsfw_3.hash`

Total download size: ~50MB





