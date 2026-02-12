# ComfyUI-LightOnOCR-2

ComfyUI nodes for [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B), LightOn's 1B-parameter OCR vision-language model. Extracts clean text from documents, receipts, tables, and more.

## Installation

1. Clone or copy this repo into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/MennilTossFlykune/ComfyUI-LightOnOCR-2.git
```

2. Install the required dependencies (from your ComfyUI Python environment):

```bash
pip install "transformers>=5.0.0" pillow
```

3. Restart ComfyUI.

## Nodes

Both nodes appear under the **LightOnOCR-2** category.

### LightOnOCR-2 Model Loader

Loads the model and processor. Connect its output to the Run node.

| Input | Type | Description |
|-------|------|-------------|
| `model_name` | combo | Model variant to load (default: `lightonai/LightOnOCR-2-1B`) |
| `device` | combo | `auto`, `cuda`, or `cpu` |
| `dtype` | combo | `auto`, `bfloat16`, `float16`, or `float32` |

### LightOnOCR-2 Run

Runs OCR on an image and outputs the extracted text.

| Input | Type | Description |
|-------|------|-------------|
| `model` | LIGHTONOCR2_MODEL | Output from the Model Loader node |
| `image` | IMAGE | Any ComfyUI image |
| `max_tokens` | INT | Maximum output tokens (default: 1024, range: 64-8192) |
| `seed` | INT | Random seed for reproducibility (default: 42) |
| `do_sample` | BOOLEAN | Enable sampling; when off, uses greedy decoding (default: True) |
| `temperature` | FLOAT | Sampling temperature (default: 0.2, range: 0.01-2.0) |
| `top_p` | FLOAT | Top-p / nucleus sampling threshold (default: 0.9, range: 0.0-1.0) |

| Output | Type |
|--------|------|
| `text` | STRING |

## Supported Model Variants

- `lightonai/LightOnOCR-2-1B` — Best OCR model
- `lightonai/LightOnOCR-2-1B-base` — Base model for fine-tuning
- `lightonai/LightOnOCR-2-1B-bbox` — OCR with image bounding boxes
- `lightonai/LightOnOCR-2-1B-bbox-base` — Base bbox model for fine-tuning
- `lightonai/LightOnOCR-2-1B-ocr-soup` — Merged variant for extra robustness
- `lightonai/LightOnOCR-2-1B-bbox-soup` — Merged OCR + bbox variant
