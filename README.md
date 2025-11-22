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
- `hyperswap_1c_256` ⭐ Recommended - best quality/speed
- `inswapper_128_fp16` - Fastest for RTX GPUs
- `simswap_unofficial_512` - Highest quality

### pixel_boost
- `256x256` - Fast, basic quality
- `512x512` ⭐ Recommended - good balance
- `768x768` - Better quality, slower
- `1024x1024` - Best quality, slowest

### face_mask_blur
- `0.0-1.0` - Controls edge blending
- `0.3` ⭐ Default - natural blending

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
