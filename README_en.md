# GoshuinScan-OSS
[日本語](README.md)

AI-Powered Digital Archiving Tool for Goshuin (御朱印)

## Features
Process Goshuin photos easily using a Python GUI:
1. **Single Image / Folder Batch Processing**: You can select either one image or an input folder. If a folder is selected, all supported images in that folder are processed automatically.
2. **Perspective Correction and Cropping**: Uses RMBG-2.0 to detect page contours and automatically corrects 3D perspective distortions to extract a perfectly flat rectangular page.
3. **Document Enhancement**: Combines docTR with classic algorithms (CLAHE and unsharp masking) to fix remaining minor alignment issues and boost contrast.
4. **Background Removal & Ink Extraction**: Re-applies RMBG-2.0 combined with global and adaptive thresholding to precisely isolate red stamps and black brush strokes, outputting a highly accurate transparent PNG.

## Requirements
- Python 3.10+
- Windows / Linux / macOS
- NVIDIA GPU highly recommended (CUDA is used automatically if available)

## Installation
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> The script will download docTR and RMBG-2.0 model weights upon first run. Ensure you have a stable internet connection.

## Usage
```bash
python app.py
```

## LoRA Model Config (.env)
If you want to use AI recognition (LoRA), configure paths in the project-root `.env` file.

```powershell
Copy-Item .env.example .env
```

Example `.env`:

```env
LORA_MODEL_PATH=K:\Qwen3-VL-4B-Instruct
LORA_ADAPTER_PATH=K:\qwen3vl-train\output\goshuin_lora_v1
```

## Hugging Face Authentication (Required for RMBG-2.0)
`briaai/RMBG-2.0` is a gated model, meaning you need to agree to their terms and request access.

1. Open this link and request access: `https://huggingface.co/briaai/RMBG-2.0`
2. Log in using the `hf` CLI (recommended):

```powershell
.\.venv\Scripts\hf auth login
```

> If you are using a **fine-grained token**, make sure to enable `Read access to public gated repositories you can access` in your token settings. Otherwise, you will encounter a `403 Forbidden` error.

Alternatively, you can provide your token via an environment variable:

```powershell
$env:HF_TOKEN = "hf_xxx"
```

If you haven't been granted access to `RMBG-2.0` yet, you can temporarily switch to an older, public model:

```powershell
$env:RMBG_MODEL_ID = "briaai/RMBG-1.4"
```

## How to Use the GUI
1. Choose input (one of the following):
   - `画像を選択` (Select Image) for single-image processing
   - `画像フォルダー` -> `フォルダーを選択` for folder batch processing
2. Choose output by `出力フォルダー` -> `フォルダーを選択`.
3. Optional: enable `GPU (CUDA) を使用` and `AI 識別 (LoRA モデル)`.
4. Click `処理開始` (Start Processing).

Supported file extensions: `.jpg .jpeg .png .bmp .webp .tif .tiff`

After processing is complete, the following files will be generated in your output directory:
- `*_enhanced_doctr.png`: The flattened, perspective-corrected and enhanced image.
- `*_ink_stamp_transparent.png`: A transparent PNG retaining only the black brush ink and red stamps.
