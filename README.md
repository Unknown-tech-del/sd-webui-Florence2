# Vision Model Companion WebUI Extension

A premium integration of local vision foundation models, including Microsoft's **Florence-2** and Alibaba's **Qwen-VL**, as a custom **Vision** tab inside AUTOMATIC1111 Stable Diffusion WebUI.

This extension runs completely locally, allowing you to run powerful vision tasks including descriptive captioning, object detection, phrase grounding, segmentations, OCR, and smart inpaint mask generation, as well as bulk pre-process image datasets for LoRA training.

---

## Key Features

- **🌐 Diverse Vision Tasks:** 
  - **Caption / Detailed Caption / More Detailed Caption:** High-quality natural language image descriptions.
  - **Object Detection:** Locate objects with boundary boxes and labels.
  - **Dense Region Caption:** Get captions matching specific coordinates in the image.
  - **Phrase Grounding:** Supply a text prompt and visually locate matches (e.g., "red car", "wooden chair").
  - **Referring Expression Segmentation:** Segment specific objects into precise polygon regions.
  - **OCR / OCR with Region:** Detect and extract text from images.
  
- **🎨 Smart Inpaint Mask Generator:** 
  - For object detection, grounding, or segmentation tasks, the extension automatically generates a pixel-aligned **black-and-white mask** of the detected regions.
  - Download or load the mask immediately into `img2img` inpainting workflows!

- **⚡ Gradio-to-Prompt Integration:**
  - Dedicated buttons to send generated descriptions directly into the **txt2img** or **img2img** prompt boxes and automatically switch tabs!

- **📂 Bulk Dataset Captioning (LoRA Prep):**
  - Run high-speed batch captioning across a local directory of images.
  - Save captions to matching `.txt` files next to each image, with support for prepending trigger words or appending custom tags.

---

## Installation

1. Open your stable-diffusion-webui.
2. Go to the **Extensions** tab.
3. Choose **Install from URL** and paste this repository's link.
4. Click **Install**.
5. Restart the WebUI completely. Dependencies (`transformers`, `timm`, `einops`, `accelerate`) will be verified and installed on startup automatically.

---

## Model Cache Configuration

By default, the model weights will be downloaded to `models/Florence2/<model_name>` within your main Stable Diffusion WebUI directory. This allows for clean filesystem management and offline operation once the snapshots are downloaded.

### Supported Default Models:
- **Official Models**:
  - `microsoft/Florence-2-base`
  - `microsoft/Florence-2-base-ft`
  - `microsoft/Florence-2-large`
  - `microsoft/Florence-2-large-ft`
  - `HuggingFaceM4/Florence-2-DocVQA`
- **Fine-Tuned Models**:
  - `MiaoshouAI/Florence-2-base-PromptGen-v1.5`
  - `MiaoshouAI/Florence-2-large-PromptGen-v1.5`
  - `MiaoshouAI/Florence-2-base-PromptGen-v2.0`
  - `MiaoshouAI/Florence-2-large-PromptGen-v2.0`
  - `thwri/CogFlorence-2.2-Large`
  - `gokaygokay/Florence-2-SD3-Captioner`
  - `gokaygokay/Florence-2-Flux-Large`
  - `NikshepShetty/Florence-2-pixelpros`
- **Alibaba Qwen-VL Models**:
  - `Qwen/Qwen3-VL-2B-Instruct`
  - `Qwen/Qwen3-VL-2B-Thinking`
  - `Qwen/Qwen2.5-VL-3B-Instruct`
  - `Qwen/Qwen2.5-VL-7B-Instruct`
  - `Qwen/Qwen2-VL-2B-Instruct`
  - `Qwen/Qwen2-VL-7B-Instruct`