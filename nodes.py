import torch
import numpy as np
from PIL import Image

try:
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
except ImportError:
    raise ImportError(
        "LightOnOCR-2 requires transformers >= 5.0.0. "
        "Please upgrade: pip install \"transformers>=5.0.0\""
    )


MODEL_VARIANTS = [
    "lightonai/LightOnOCR-2-1B",
    "lightonai/LightOnOCR-2-1B-base",
    "lightonai/LightOnOCR-2-1B-bbox",
    "lightonai/LightOnOCR-2-1B-bbox-base",
    "lightonai/LightOnOCR-2-1B-ocr-soup",
    "lightonai/LightOnOCR-2-1B-bbox-soup",
]


class LightOnOCR2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (MODEL_VARIANTS, {"default": "lightonai/LightOnOCR-2-1B"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("LIGHTONOCR2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "LightOnOCR-2"

    def load_model(self, model_name, device, dtype):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        dtype_map = {
            "auto": torch.float32 if device == "mps" else torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        model = LightOnOcrForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(device)
        processor = LightOnOcrProcessor.from_pretrained(model_name)

        return ({"model": model, "processor": processor, "device": device, "dtype": torch_dtype},)


class LightOnOCR2Run:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LIGHTONOCR2_MODEL",),
                "image": ("IMAGE",),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run_ocr"
    CATEGORY = "LightOnOCR-2"

    def run_ocr(self, model, image, max_tokens):
        ocr_model = model["model"]
        processor = model["processor"]
        device = model["device"]
        torch_dtype = model["dtype"]

        # Convert ComfyUI image tensor (B,H,W,C) float [0,1] to PIL
        img_array = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_array)

        conversation = [
            {
                "role": "user",
                "content": [{"type": "image", "image": pil_image}],
            }
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device=device, dtype=torch_dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }

        output_ids = ocr_model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)

        return (output_text,)
