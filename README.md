# FaceFusion ComfyUI (Unofficial)

![FaceFusion ComfyUI Demo](assets/Timeline%201.gif)

Advanced face swapping for ComfyUI with **local ONNX inference** - no API required!

## 🚀 Quick Start

### Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/huygiatrng/Facefusion_comfyui.git
cd Facefusion_comfyui
pip install -r requirements.txt
```

Restart ComfyUI. Nodes will appear under `FaceFusion` and `FaceFusion API`.

### Basic Usage

1. Add "Load Image" nodes for source (face) and target (body) images
2. Add **"FF: Advanced Swap Face (Image)"** node
3. Connect images and set `api_token` to `-1` (local mode - default)
4. Choose model: `hyperswap_1c_256` (recommended)
5. Connect to "Preview Image" and run!

First run downloads models (~200MB), then everything runs locally.

---

## 📋 Main Nodes

### Face Swapping

- **SwapFaceImage** - Basic image face swap
- **AdvancedSwapFaceImage** ⭐ - Full control with all options (recommended)
- **AdvancedSwapFaceVideo** - Video face swapping with parallel processing
- **DeepSwapFaceImage** - DeepFaceLive `.dfm` model support (single images and batches)
- **FaceSwapApplier** - Swap specific detected faces

### Detection & Tools

- **FaceDetectorNode** - Detect and analyze faces
- **FaceDataVisualizer** - Debug tool showing detected faces
- **PixelBoostNode** - Configure pixel boost settings

---

## ⚙️ Key Parameters

### api_token
- `-1` = Local inference (default) ✅ No internet required
- `your_token` = API mode (requires internet)

### face_swapper_model
**13 models available** - All auto-download on first use:
- `hyperswap_1c_256` ⭐ Recommended - best quality/speed
- `hyperswap_1a_256`, `hyperswap_1b_256` - HyperSwap variants
- `ghost_1_256`, `ghost_2_256`, `ghost_3_256` - Apache-2.0 license (commercial OK)
- `hififace_unofficial_256` - High fidelity faces
- `inswapper_128_fp16` ⚡ Fastest for RTX GPUs
- `inswapper_128` - Standard InsightFace
- `blendswap_256` - Good blending quality
- `simswap_256`, `simswap_unofficial_512` - SimSwap variants
- `uniface_256` - Uniform face handling

### pixel_boost
- `256x256` - Fast, basic quality
- `512x512` ⭐ Recommended - good balance
- `768x768` - Better quality, slower
- `1024x1024` - Best quality, slowest

### face_mask_blur
- `0.0-1.0` - Controls edge blending
- `0.3` ⭐ Default - natural blending

### Face Mask Types (Multi-Select)

You can enable **multiple mask types** at once! Masks are combined to create precise face boundaries.

| Mask Type | Option | Description |
|-----------|--------|-------------|
| **Box** | `use_box_mask` ✅ | Rectangular mask with blur around face edges (default ON) |
| **Occlusion** | `use_occlusion_mask` | Detects occlusions (hands, hair, objects covering face) - requires `face_occluder_model` |
| **Area** | `use_area_mask` | Masks specific face areas using landmarks |
| **Region** | `use_region_mask` | Semantic segmentation of face parts - requires `face_parser_model` |

#### face_mask_areas (for Area mask)
Comma-separated list of areas:
- `upper-face` - Forehead and upper face
- `lower-face` - Chin and jaw area  
- `mouth` - Mouth region only
- Example: `upper-face,lower-face,mouth` (default - full face)

#### face_mask_regions (for Region mask)
Comma-separated list of regions:
- `skin` - Face skin only
- `left-eyebrow`, `right-eyebrow` - Eyebrows
- `left-eye`, `right-eye` - Eyes
- `glasses` - Glasses area
- `nose` - Nose
- `mouth` - Mouth area
- `upper-lip`, `lower-lip` - Lips
- Example: `skin,nose,mouth,upper-lip,lower-lip` (default)

#### face_mask_padding
Edge padding for box mask: `top,right,bottom,left`
- Example: `5,5,5,5` - 5% padding on all sides
- Default: `0,0,0,0`

#### Recommended Mask Combinations

| Use Case | Masks | Notes |
|----------|-------|-------|
| **Standard swap** | Box only ✅ | Fast, good for most cases |
| **Hands/objects near face** | Box + Occlusion | Preserves hands covering face |
| **Precise face boundary** | Box + Region | Better edge handling with hair |
| **Mouth preservation** | Box + Region (`skin,nose`) | Keeps original mouth |
| **Full quality** | Box + Occlusion + Region | Best quality, slower |

### face_selector_mode
- `one` - Single face (use face_position to select)
- `many` - All detected faces
- `reference` - Match faces similar to reference image

### sort_order
- `large-small` ⭐ Biggest face first
- `left-right`, `top-bottom` - Spatial sorting
- `best-worst` - By detection confidence

---

## 🎯 Example Workflows

### Simple Swap
```
Source Image → Advanced Swap Face ← Target Image → Preview
             (api_token: -1)
```

### Batch Processing (Multiple Images)
```
Source Image → Advanced Swap Face ← Load Image Batch → Preview
             (automatically processes all)
```

### With Face Detection
```
Target → Face Detector → Visualize (debug)
              ↓
Source → Face Swap Applier → Preview
```

### Video Swap
```
Source Image → Advanced Swap Video ← Target Video
             (max_workers: 8)
                    ↓
               Save Video
```

### Smart Batch Handling

All image swapper nodes **automatically detect and handle**:
- ✅ Single image (shape: [1, H, W, 3])
- ✅ Batch of images (shape: [N, H, W, 3])
- ✅ Image lists from Load Image Batch nodes
- ✅ Returns same format as input

**Example:** Feed 10 images → Get 10 swapped images back!

### DeepFaceLive DFM models

1. Create or download a DeepFaceLive-compatible `.dfm` model.
2. Place it in `models/deep_swapper/` (subfolders are supported).
3. Restart ComfyUI so the model appears in **FF: Deep Swap Face (DFM)**.
4. Use `morph` when the selected DFM model exposes a morph input. Models
   without that input safely ignore the value.

DFM models are identity-specific, so this node does not require a source image.

### EXR and HDR image sequences

Local image and video swapping now uses a clipped SDR proxy only for ONNX
inference, then applies the swap delta back to the original float tensor.
Negative values and highlights above `1.0` therefore survive unchanged outside
the swapped region instead of wrapping to black during `uint8` conversion.

---

## 🔧 Common Settings

### For Speed
- Model: `inswapper_128_fp16`
- Pixel Boost: `256x256` or `512x512`
- GPU with CUDA enabled

### For Quality
- Model: `hyperswap_1c_256` or `simswap_unofficial_512`
- Pixel Boost: `768x768` or `1024x1024`
- Blur: `0.3-0.5`

### For Video
- Model: `hyperswap_1c_256`
- Pixel Boost: `512x512`
- Max Workers: `4-8`

---

## 🛠️ Troubleshooting

### No Faces Detected
- Lower `score_threshold` to 0.3-0.4
- Check image quality and lighting
- Ensure face is clearly visible

### Out of Memory
- Lower `pixel_boost` (256×256 or 512×512)
- Use smaller model (`inswapper_128_fp16`)
- Process fewer faces (`mode='one'`)

### Slow Performance
- Enable GPU/CUDA
- Use faster model (`inswapper_128_fp16`)
- Lower pixel boost resolution

### Models Won't Download
- Check internet connection
- Verify disk space (~500MB per model)
- Manual download: https://github.com/facefusion/facefusion-assets/releases/

---

## 📦 Models

Models auto-download to: `custom_nodes/Facefusion_comfyui/models/`

Available models (~100-500MB each):
- hyperswap_1a/1b/1c_256
- inswapper_128, inswapper_128_fp16
- blendswap_256, simswap_256, simswap_unofficial_512
- uniface_256

Face detection: scrfd_2.5g (~3MB), arcface_w600k_r50 (~166MB)

---

## 🎓 Tips

- **Start with defaults** - They work well for most cases
- **Use local mode** (api_token: -1) - It's faster and private
- **GPU makes a huge difference** - 10-50× faster than CPU
- **Adjust blur** - Higher values (0.4-0.6) for smoother blending
- **Match angles** - Source and target faces should face similar directions
- **Batch processing** - Feed multiple images at once, get all results automatically
- **Use Load Image Batch** - Perfect for processing folders of images

### Mask Selection Tips
- **Hands near face?** → Enable `use_occlusion_mask` with `face_occluder_model: xseg_1`
- **Hair blending issues?** → Enable `use_region_mask` with `face_parser_model: bisenet_resnet_34`
- **Want original mouth?** → Use Region mask with `face_mask_regions: skin,nose,left-eye,right-eye`
- **Processing speed priority?** → Use only Box mask (default)

---

## 📝 Local vs API

| Feature | Local (api_token: -1) | API (with token) |
|---------|----------------------|------------------|
| Internet | Not required | Required |
| Speed | Fast with GPU | Depends on connection |
| Privacy | Complete | Processed remotely |
| Cost | Free | May have limits/costs |
| Quality | Full pixel boost | Limited options |

**Recommendation:** Use local mode (default) for best results!

---

## 🔗 Links

- **FaceFusion**: https://github.com/facefusion/facefusion
- **Models**: https://github.com/facefusion/facefusion-assets/releases/
- **API**: https://facefusion.io (optional)

---

## 📄 License

Respect model licenses:
- InsightFace models: Non-commercial use
- Face swapper models: Check vendor licenses

---

## 🆘 Support

- Report issues on GitHub
- Check console output for errors
- Enable debug mode by uncommenting print statements in code

---

**Happy Face Swapping! 🎭✨**
