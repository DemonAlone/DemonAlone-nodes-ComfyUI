import comfy.samplers
import folder_paths 
import os
import torch
import torch.nn.functional as F
import comfy
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import nodes
import json
import hashlib
import fnmatch
from PIL.JpegImagePlugin import JpegImageFile  
from pathlib import Path
from datetime import datetime
from PIL.ExifTags import TAGS, GPSTAGS, IFD         
from PIL.PngImagePlugin import PngImageFile
from comfy_execution.graph_utils import GraphBuilder
from comfy_execution.graph import ExecutionBlocker
import re
import torchvision.transforms as transforms

def get_sampler_list():
    return ["none"] + comfy.samplers.KSampler.SAMPLERS

def get_scheduler_list():
    return ["none"] + comfy.samplers.KSampler.SCHEDULERS

def get_model_list():
    checkpoints = folder_paths.get_filename_list("checkpoints")
    return ["none"] + checkpoints

def get_diffusion_model_file_list():
    diffusion_models = folder_paths.get_filename_list("diffusion_models")
    return ["none"] + diffusion_models
    
def get_checkpoint_list():
    """Returns a list of checkpoint files plus 'none'."""
    checkpoints = folder_paths.get_filename_list("checkpoints")
    return ["none"] + checkpoints

def get_vae_list():
    """Returns a list of VAE files plus 'none'."""
    vae_files = folder_paths.get_filename_list("vae")   # key "vae"
    return ["none"] + vae_files
    
def get_text_encoder_list():
    return ["none"] + folder_paths.get_filename_list("text_encoders")

def get_lora_list():
    """
    Return a list of all LoRA files located in the `loras/` folder,
    prepended with two placeholder entries:
    `"none"`       – used when nothing is chosen, simply skips this slot
    `"Empty_value"`– used during tests that need an empty / space result
    The full path from the folder‑search will be stored in the dropdowns.
    """
    loras = folder_paths.get_filename_list("loras")  # full path
    return ["none", "Empty_value"] + loras
    
def build_metadata(image_path: str):
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"File not found: {image_path}")

    img = Image.open(image_path)

    # 1. Basic file information -----------------------------------
    fileinfo = {
        "filename": str(Path(image_path)),
        "resolution": f"{img.width}x{img.height}",
        "date": datetime.fromtimestamp(os.path.getmtime(image_path)).isoformat(),
        "size": str(round(os.path.getsize(image_path) / 1024**2, 2)) + " MB",
    }
    metadata = {"fileinfo": fileinfo}
    prompt   = {}

    # PNG metadata (img.info)
    if isinstance(img, PngImageFile):
        for k, v in img.info.items():
            try:
                # If the user commented arbitrary fields as JSON, parse them.
                metadata[k] = json.loads(v)
            except Exception:
                metadata[k] = str(v)

        # PNG usually contains prompt / workflow / parameters
        if "prompt" in metadata:
            try:
                # ComfyUI stores this as a string, but the `raw` node already parses it as JSON and places it in the `prompt` key.
                prompt.update(metadata["prompt"])
            except Exception:
                pass

        if "workflow" in metadata:
            try:
                metadata["workflow"] = json.loads(metadata["workflow"])
            except Exception:
                pass

        # In the 'raw' node, workflow is stored as a Python dict (not in `metadata`),
        # but we put it here so that it appears in the output JSON.
        prompt_from_image = prompt.update(metadata.get("prompt", {}))
        if "workflow" in metadata:
            try:
                metadata["workflow"] = json.loads(metadata["workflow"])
            except Exception:
                pass

    # JPEG EXIF
    elif isinstance(img, JpegImageFile):
        exif = img.getexif()
        for k, v in exif.items():
            tag = TAGS.get(k, k)
            metadata[str(tag)] = str(v)

    # WebP EXIF (prompt / workflow)
    if img.format == "WEBP":
        try:
            import piexif
            exif_data = piexif.load(image_path)
            prompt_from_webp = exif_data.get('0th', {}).get(271, None)
            if prompt_from_webp is not None:
                raw_prompt = prompt_from_webp.decode("utf-8").replace("Prompt:", "", 1)
                try:
                    metadata["prompt"] = json.loads(raw_prompt)
                except Exception:
                    pass

            workflow_data = exif_data.get('0th', {}).get(270, None)
            if workflow_data is not None:
                raw_workflow = workflow_data.decode("utf-8").replace("Workflow:", "", 1)
                try:
                    metadata["workflow"] = json.loads(raw_workflow)
                except Exception:
                    pass

        except ValueError:   # piexif could not read
            logger.warn("piexif error on WebP – ignore")

    return img, prompt, metadata
    
class SamplerGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        samplers = get_sampler_list()
        inputs = {"required": {f"sampler_{i+1}": (samplers, {"default": "none"}) for i in range(10)}}
        return inputs
    RETURN_TYPES = ("STRING", "LIST")  
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"sampler_{i+1}")
            if name and name != "none": selected.append(name)
        string_output = ", ".join(selected)
        list_output = selected  
        return (string_output, list_output)

class SchedulerGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        schedulers = get_scheduler_list()
        inputs = {"required": {f"scheduler_{i+1}": (schedulers, {"default": "none"}) for i in range(10)}}
        return inputs
    RETURN_TYPES = ("STRING", "LIST")
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"scheduler_{i+1}")
            if name and name != "none": selected.append(name)
        string_output = ", ".join(selected)
        list_output = selected  
        return (string_output, list_output)

class ModelGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list()
        inputs = {"required": {f"model_{i+1}": (models, {"default": "none"}) for i in range(10)}}
        return inputs
    RETURN_TYPES = ("STRING","LIST")
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"model_{i+1}")
            if name and name != "none": selected.append(name)
        string_output = ", ".join(selected)
        list_output = selected  
        return (string_output, list_output)

class DiffusionModelGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_diffusion_model_file_list()
        inputs = {"required": {f"diff_model_{i+1}": (models, {"default": "none"}) for i in range(10)}} 
        return inputs
    RETURN_TYPES = ("STRING","LIST")
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"diff_model_{i+1}")
            if name and name != "none": selected.append(name)
        string_output = ", ".join(selected)
        list_output = selected  
        return (string_output, list_output)

class AnyAdapterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"input_any": ("*",)}}

    RETURN_TYPES = ("*",)
    FUNCTION = "adapt"
    CATEGORY = "utils"

    def adapt(self, input_any=None):
        """
        If the input is not connected, `input_any` will be None.
        In that case we simply return nothing (or None) so that the node
        passes an empty result downstream.
        """
        if input_any is None:
            # Returning a single element – ComfyUI expects at least one output value.
            return (None,)
        return (input_any,)

class CheckpointSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Input: a single checkpoint. The combo‑menu will be built automatically.
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    # First element in the tuple → combo, second → STRING
    RETURN_TYPES = (
        folder_paths.get_filename_list("checkpoints"),
        "STRING",
    )
    RETURN_NAMES = ("ckpt_name", "ckpt_name_str")
    CATEGORY = "utils"   # Folder in UI

    FUNCTION = "get_ckpt_name"

    def get_ckpt_name(self, ckpt_name):
        return ckpt_name, ckpt_name

class DiffusionModelSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Input: one file from the diffusion_models folder (combo‑menu)
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                ),
            }
        }

    # First element → combo, second → STRING
    RETURN_TYPES = (
        folder_paths.get_filename_list("diffusion_models"),
        "STRING",
    )
    RETURN_NAMES = ("model_name", "model_name_str")
    CATEGORY = "utils"     # Folder in UI

    FUNCTION = "get_model"

    def get_model(self, model_name):
        return model_name, model_name

class VAEGeneratorNode:
    """
    Generator of a list of VAE files.
    Parameters: 10 dropdowns → string of the selected values.
    """
    @classmethod
    def INPUT_TYPES(cls):
        vaes = get_vae_list()
        inputs = {
            "required": {f"vae_{i+1}": (vaes, {"default": "none"}) for i in range(10)}
        }
        return inputs
    RETURN_TYPES = ("STRING","LIST")
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"vae_{i+1}")
            if name and name != "none":
                selected.append(name)
        string_output = ", ".join(selected)
        list_output = selected  
        return (string_output, list_output)

class TextEncoderGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        encoders = get_text_encoder_list()
        inputs = {
            "required": {f"text_enc_{i+1}": (encoders, {"default": "none"}) for i in range(10)}
        }
        return inputs

    RETURN_TYPES = ("STRING","LIST")
    FUNCTION = "generate_string"
    CATEGORY = "utils"

    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"text_enc_{i+1}")
            if name and name != "none":
                selected.append(name)
        string_output = ", ".join(selected)
        list_output = selected  
        return (string_output, list_output)

class VAESelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Input: one VAE. Combo‑menu will be built from files in the “vae” folder.
                "vae_name": (
                    folder_paths.get_filename_list("vae"),
                ),
            }
        }

    # First output – combo (can be connected to Load VAE, etc.)
    # Second – string with the same name
    RETURN_TYPES = (
        folder_paths.get_filename_list("vae"),  # combo‑output
        "STRING",                               # plain text output
    )
    RETURN_NAMES = ("vae_name", "vae_name_str")
    CATEGORY = "utils"     # subfolder in UI

    FUNCTION = "get_vae"

    def get_vae(self, vae_name):
        return vae_name, vae_name

class TextEncoderSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Input: one Text‑Encoder. Combo‑menu will be built from files in the “text_encoders” folder.
                "enc_name": (
                    folder_paths.get_filename_list("text_encoders"),
                ),
            }
        }

    RETURN_TYPES = (
        folder_paths.get_filename_list("text_encoders"),  # combo‑output
        "STRING",                                        # plain text output
    )
    RETURN_NAMES = ("enc_name", "enc_name_str")
    CATEGORY = "utils"

    FUNCTION = "get_encoder"

    def get_encoder(self, enc_name):
        return enc_name, enc_name

class StringToIntNode:
    """
    Accepts a string and attempts to convert it into an integer.
    If conversion fails – returns 0 (or you could raise an error instead).
    """
    @classmethod
    def INPUT_TYPES(cls):
        # "STRING" guarantees that the user sees a text input field
        return {"required": {"text_value": ("STRING",)}}

    RETURN_TYPES = ("INT",)            # output – integer
    FUNCTION = "convert"
    CATEGORY = "utils"

    def convert(self, text_value):
        try:
            return (int(text_value),)
        except Exception as e:
            # Log the error in the console; here we simply return 0
            print(f"[StringToIntNode] Conversion error for '{text_value}': {e}")
            return (0,)

class StringToFloatNode:
    """
    Accepts a string and attempts to convert it into a floating‑point number.
    If conversion fails – returns 0.0.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text_value": ("STRING",)}}

    RETURN_TYPES = ("FLOAT",)          # output – float
    FUNCTION = "convert"
    CATEGORY = "utils"

    def convert(self, text_value):
        try:
            return (float(text_value),)
        except Exception as e:
            print(f"[StringToFloatNode] Conversion error for '{text_value}': {e}")
            return (0.0,)

class TextConcatNode:
       # modified node TextConcat from https://github.com/bash-j/mikey_nodes
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"delimiter": ("STRING", {"default": " "})},
            "optional": {f"text{i}": ("STRING", {"default": ""})
                         for i in range(1, 6)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat"
    CATEGORY = "utils"

    def concat(self,
               delimiter,
               text1="", text2="", text3="",
               text4="", text5=""):
        """Collect non‑empty strings into a single text."""
        texts = [t for t in (text1, text2, text3, text4, text5) if t]
        return (delimiter.join(texts),)     

class LORASelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        loras = get_lora_list()
        inputs = {"required": {}}

        for i in range(10):
            # Dropdown – full path to a LoRA file (or the placeholders)
            inputs["required"][f"lora_{i+1}"] = (loras, {"default": "none"})
            # Weight for the selected LoRA
            inputs["required"][f"weight_{i+1}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.00,
                    "max": 10.00,
                    "step": 0.01,
                },
            )
        return inputs

    # Output: a string that can be used in prompts, and a LIST of the tags
    RETURN_TYPES = ("STRING", "LIST")
    FUNCTION = "generate_string"
    CATEGORY = "utils"

    def generate_string(self, **kwargs):
        parts = []      # will become the comma‑separated string output
        lora_list = []  # separate list that may contain empty items

        for i in range(10):
            path   = kwargs.get(f"lora_{i+1}")
            weight = kwargs.get(f"weight_{i+1}")

            # --- 0. Nothing selected ---------------------------------------
            if not path or path == "none":
                continue

            # --- 1. Special “Empty_value” item -----------------------------
            if path == "Empty_value":
                # Insert an empty string / space (no actual LoRA tag will appear)
                parts.append("")          # nothing is added to the final string
                lora_list.append("")      # but a placeholder remains in the list
                continue

            # --- 2. Normal LoRA --------------------------------------------
            name_without_ext = os.path.splitext(os.path.basename(path))[0]
            weight_str = f"{float(weight):.2f}"
            lora_tag = f"<lora:{name_without_ext}:{weight_str}>"

            parts.append(lora_tag)
            lora_list.append(lora_tag)

        string_output = ", ".join(parts).strip()
        return (string_output, lora_list)

class ClipSkipSliderNode:
    """
    Emits an integer in the range [-24 … -1].
    The slider is bounded by its min/max attributes – the user cannot
    select a value outside this interval.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 1. Input name → type INT
                # 2. Property dictionary → default, min, max
                "value": ("INT", {"default": -1, "min": -24, "max": -1})
            }
        }

    RETURN_TYPES = ("INT",)          # single output – integer
    RETURN_NAMES = ("value",)        # optional – gives the output a name
    FUNCTION = "get_value"           # method that will be invoked
    CATEGORY = "utils"               # subfolder in UI

    def get_value(self, value):
        """
        Receives the slider's current integer value and returns it.
        The return is wrapped in a tuple because the node interface expects
        an iterable of outputs.
        """
        return (value,)

class PonyPrefixesNode:
    
    
    """
    
    Score     – 5 variants
        "-"               → None
        "Everything"      → "score_9, score_8_up, score_7_up, score_6_up, score_5_up, "
        "Average"         → "score_9, score_8_up, score_7_up, score_6_up, score_5_up, "
        "Good"            → "score_9, score_8_up, score_7_up, "
        "Only the best"   → "score_9, "

    Rating    – 4 variants
        "-"               → None
        "Safe"            → "rating_safe, "
        "Questionable"    → "rating_questionable, "
        "Explicit"        → "rating_explicit, "

    Source    – 5 variants
        "-"               → None
        "Anime"           → "source_anime, "
        "Furry"           → "source_furry, "
        "Cartoon"         → "source_cartoon, "
        "Pony"            → "source_pony, "

   In results combined string (order: Score → Rating → Source).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 1 list – Score
                "score": (
                    [
                        "-",
                        "Everything",
                        "Average",
                        "Good",
                        "Only the best",
                    ],
                ),
                # 2 list – Rating
                "rating": (
                    [
                        "-",
                        "Safe",
                        "Questionable",
                        "Explicit",
                    ],
                ),
                # 3 list – Source
                "source": (
                    [
                        "-",
                        "Anime",
                        "Furry",
                        "Cartoon",
                        "Pony",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_string",)   # output name
    FUNCTION = "generate"
    CATEGORY = "utils"

    # mappings
    _SCORE_MAP = {
        "Everything":     "score_9, score_8_up, score_7_up, score_6_up, score_5_up, ",
        "Average":        "score_9, score_8_up, score_7_up, score_6_up, score_5_up, ",
        "Good":           "score_9, score_8_up, score_7_up, ",
        "Only the best":  "score_9, ",
    }

    _RATING_MAP = {
        "Safe":          "rating_safe, ",
        "Questionable":  "rating_questionable, ",
        "Explicit":      "rating_explicit, ",
    }

    _SOURCE_MAP = {
        "Anime":   "source_anime, ",
        "Furry":   "source_furry, ",
        "Cartoon": "source_cartoon, ",
        "Pony":    "source_pony, ",
    }

    def generate(self, score="-", rating="-", source="-"):
        """Comnining whole string"""
        parts = []

        if score != "-":
            parts.append(self._SCORE_MAP.get(score, ""))

        if rating != "-":
            parts.append(self._RATING_MAP.get(rating, ""))

        if source != "-":
            parts.append(self._SOURCE_MAP.get(source, ""))

        result = "".join(parts)

        return (result,)

class ImageResizeNode:

    # ImageResizeNode is based on 🔧 Image Resize from Efficiency Nodes
    """
    # Efficiency Nodes - A collection of my ComfyUI custom nodes to help streamline workflows and reduce total node count.
    # by Luciano Cirino (Discord: TSC#9184) - April 2023 - October 2023
    # https://github.com/LucianoCirino/efficiency-nodes-comfyui
    Resize an image and a mask synchronously.
    The mask is resized with nearest‑neighbor to keep its binary nature.
    """

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image_out", "mask_out", "width", "height")
    FUNCTION     = "execute"
    CATEGORY     = "utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {          # both inputs are now optional
                "image":  ("IMAGE",),
                "mask":   ("MASK",),

                "width":  ("INT", {"default": 512, "min": 0, "max": 16834}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16834}),

                "method":        (["stretch",
                                   "keep proportion",
                                   "fill / crop",
                                   "pad"],),
                "interpolation": (["nearest",
                                   "bilinear",
                                   "bicubic",
                                   "area",
                                   "nearest-exact",
                                   "lanczos"],),
                "condition":     (["always",
                                   "downscale if bigger",
                                   "upscale if smaller",
                                   "if bigger area",
                                   "if smaller area"],),
            }
        }

    def execute(self,
                image=None,
                mask=None,
                width: int = 512,
                height: int = 512,
                method: str = "stretch",
                interpolation: str = "nearest",
                condition: str = "always"):
        """
        Resizes both an image and a mask (if provided) using the same target size.
        If only one of them is connected, that one will be resized while the other
        stays untouched.  If an input is missing, the corresponding output tensor
        will be empty (filled with zeros).
        """

        has_image = image is not None
        has_mask  = mask is not None

        if not (has_image or has_mask):
            raise ValueError("At least one of 'image' or 'mask' must be connected")

        # --------- 0. Determine original sizes ----------
        source_tensor = image if has_image else mask
        if source_tensor.ndim == 4:
            _, oh, ow, _ = source_tensor.shape   # (B,H,W,C)
        elif source_tensor.ndim == 3:
            _, oh, ow = source_tensor.shape      # (B,H,W)
        else:
            raise ValueError(f"Unsupported source tensor shape: {source_tensor.shape}")

        # --------- 1. Compute target size ----------
        pad_left = pad_right = pad_top = pad_bottom = 0
        x = y = x2 = y2 = None

        if method == "keep proportion":
            ratio   = min(width / ow if width else float("inf"),
                         height / oh if height else float("inf"))
            new_w, new_h = round(ow * ratio), round(oh * ratio)
            target_w, target_h = new_w, new_h

        elif method == "pad":
            ratio   = min(width / ow if width else float("inf"),
                         height / oh if height else float("inf"))
            new_w, new_h = round(ow * ratio), round(oh * ratio)
            pad_left  = (width - new_w) // 2
            pad_right = width - new_w - pad_left
            pad_top   = (height - new_h) // 2
            pad_bottom= height - new_h - pad_top
            target_w, target_h = new_w, new_h

        elif method == "fill / crop":
            target_w = width if width else ow
            target_h = height if height else oh
            ratio    = max(target_w / ow, target_h / oh)
            new_w, new_h = round(ow * ratio), round(oh * ratio)

            x  = (new_w - target_w) // 2
            y  = (new_h - target_h) // 2
            x2 = x + target_w
            y2 = y + target_h

            if x2 > new_w:   x -= (x2 - new_w)
            if x < 0:        x = 0
            if y2 > new_h:   y -= (y2 - new_h)
            if y < 0:        y = 0

            target_w, target_h = new_w, new_h

        else:                          # stretch or unknown method
            target_w = width  if width  else ow
            target_h = height if height else oh

        new_width, new_height = target_w, target_h

        # --------- 2. When to perform resize ----------
        should_resize = (
            condition == "always" or
            ("downscale if bigger" == condition and (oh > new_height or ow > new_width)) or
            ("upscale if smaller" == condition and (oh < new_height or ow < new_width)) or
            ("bigger area" in condition and (oh * ow > new_height * new_width)) or
            ("smaller area" in condition and (oh * ow < new_height * new_width))
        )

        # --------- 3. Resize image ----------
        if has_image:
            img = image.permute(0, 3, 1, 2)   # B,C,H,W

            if should_resize:
                if interpolation == "lanczos" and comfy is not None:
                    img = comfy.utils.lanczos(img, new_width, new_height)
                else:
                    kwargs = {"size": (new_height, new_width)}
                    if interpolation in ("linear", "bilinear", "bicubic", "trilinear"):
                        kwargs["align_corners"] = False
                    img = F.interpolate(img, mode=interpolation, **kwargs)

                if method == "pad" and (pad_left or pad_right or pad_top or pad_bottom):
                    img = F.pad(img,
                                (pad_left, pad_right, pad_top, pad_bottom),
                                mode='constant', value=0)
                if method == "fill / crop":
                    img = img[:, :, y:y2, x:x2]

            image_out = img.permute(0, 2, 3, 1)   # B,H,W,C
        else:
            # Create an empty tensor for image_out
            batch_size = source_tensor.shape[0]
            channels = 3 if has_image else 1  # If there's no image input, use one channel
            image_out = torch.zeros(batch_size, new_height, new_width, channels)

        # --------- 4. Resize mask ----------
        if has_mask:
            # --- Prepare input for processing ---
            if mask.ndim == 3:          # (B, H, W)
                msk = mask.unsqueeze(1)    # -> B,1,H,W
            elif mask.ndim == 4 and mask.shape[3] == 1:   # (B, H, W, 1)
                msk = mask.permute(0, 3, 1, 2)           # -> B,1,H,W
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")

            if should_resize:
                msk = F.interpolate(msk,
                                    size=(new_height, new_width),
                                    mode='nearest')

                if method == "pad" and (pad_left or pad_right or pad_top or pad_bottom):
                    msk = F.pad(msk,
                                (pad_left, pad_right, pad_top, pad_bottom),
                                mode='constant', value=0)
                if method == "fill / crop":
                    msk = msk[:, :, y:y2, x:x2]

            # --- Return mask in format (B,H,W) ---
            mask_out = msk.squeeze(1)      # remove channel 1
        else:
            # Create an empty tensor for mask_out
            batch_size = source_tensor.shape[0]
            mask_out = torch.zeros(batch_size, new_height, new_width)

        return image_out, mask_out, new_width, new_height

class ResizeMethodControlNode:
    """
    Remote control unit with resizing method.
    Sends the selected value as a combo type for compatibility.
    Can be connected to the 'method' input of ImageResizeNode.
    """
    #Define the list once to avoid duplication.
    METHODS = ["stretch", "keep proportion", "fill / crop", "pad"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "method": (cls.METHODS,),
            }
        }

    RETURN_TYPES = (METHODS,) 
    RETURN_NAMES = ("method",)
    FUNCTION    = "get_method"
    CATEGORY    = "utils"

    def get_method(self, method: str):
        return (method,)

class ResizeInterpolationControlNode:
    """
    Outputs the chosen interpolation type.
    Can be connected to the 'interpolation' input of ImageResizeNode.
    """
    INTERPOLATION = ["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interpolation": (cls.INTERPOLATION,),
            }
        }

    RETURN_TYPES = (INTERPOLATION,)
    RETURN_NAMES = ("interpolation",)
    FUNCTION    = "get_interpolation"
    CATEGORY    = "utils"

    def get_interpolation(self, interpolation: str):
        return (interpolation,)

class AnyConcatNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            # Delimiter: always a text field, can be changed manually
            "required": {"delimiter": ("STRING", {"default": " "})},

            # text1…text5 are only connectors.
            "optional": {f"text{i}": ("STRING", {"forceInput": True}) for i in range(1, 6)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat"
    CATEGORY = "utils"

    def concat(self, delimiter: str, **kwargs):
        """
        kwargs contains only those slots that were actually connected.
        If a slot was not connected, it simply is absent from the dict.
        """
        texts = [str(v) for v in kwargs.values() if v]
        return (delimiter.join(texts),)

class OptionalCondMergeNode:
    """
    Smart "merge" for conditioning.

    - inputs: cond1, cond2, cond3 (optional)
    - output: one merged-conditioning
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            # no required inputs – everything is optional
            "required": {},
            "optional": {
                "cond1": ("CONDITIONING",),
                "cond2": ("CONDITIONING",),
                "cond3": ("CONDITIONING",)
            }
        }

    RETURN_TYPES = ("CONDITIONING",)      # single output
    FUNCTION     = "merge"
    CATEGORY     = "conditioning"        # UI sub‑folder

    def merge(self, **kwargs):
        """
        kwargs is a dict: {'cond1': ..., 'cond2': ..., 'cond3': ...}
        If an input is not connected it will be None.
        """
        # 1️. Collect only the ones that exist
        conds = [c for c in (kwargs.get('cond1'),
                            kwargs.get('cond2'),
                            kwargs.get('cond3')) if c is not None]

        n = len(conds)
        if n == 0:
            # Node has no connections – return None.
            # When muted, ComfyUI simply ignores this output.
            return (None,)

        weight = 1.0 / n          # automatically calculated weight

        # 2️. Merge conditioning layer‑by‑layer
        # Each conditioning is a list of tuples: [(tensor, meta), ...]
        merged = []
        for layer_idx in range(len(conds[0])):

            # take the tensor from each input and multiply by weight
            tensors_for_layer = [c[layer_idx][0] * weight for c in conds]

            # sum element‑wise across all inputs
            summed_tensor = torch.sum(torch.stack(tensors_for_layer), dim=0)

            # keep metadata from the first conditioning (usually scale, etc.)
            merged.append((summed_tensor, conds[0][layer_idx][1]))

        return (merged,)

class ScaleImageAspectNode:
    # ScaleImageAspectNode is based on 🔧 Image Resize from Efficiency Nodes
    """
    # Efficiency Nodes - A collection of my ComfyUI custom nodes to help streamline workflows and reduce total node count.
    # by Luciano Cirino (Discord: TSC#9184) - April 2023 - October 2023
    # https://github.com/LucianoCirino/efficiency-nodes-comfyui
    Resize an image and a mask synchronously.
    The mask is resized with nearest‑neighbor to keep its binary nature.
    """
    """
    Resizes an image while preserving its aspect ratio.
    A single parameter `max_side` specifies the target size of **the longest** side of the image.
    If set to 0, the image is passed through unchanged.

        * If the current longest side > max_side → it is shrunk to max_side,
          the other side is scaled proportionally.
        * If the current longest side < max_side → it is enlarged to max_side
          (again by a proportional factor).

    The `max_side` value has a step of 64 and a maximum of 16384 pixels.
    """

    # ── I/O definitions for ComfyUI ───────────────────────────────

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)

    FUNCTION     = "execute"
    CATEGORY     = "utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE",),                     # (B, H, W, C)
                "max_side":   ("INT", {"default": 0,
                                         "min": 0,
                                         "max": 16384,
                                         "step": 64}),          # target size of the longest side
                "interpolation": ([
                    "nearest",
                    "bilinear",
                    "bicubic",
                    "area",
                    "nearest-exact",
                    "lanczos"
                ],),
            }
        }

    # ── Execution logic ───────────────────────────────────────────

    def execute(self,
                image=None,
                max_side: int = 0,
                interpolation: str = "nearest"):
        """
        Resizes the supplied image to a size that fits within
        `max_side` (if > 0), keeping its aspect ratio.
        If `max_side` is 0, returns the original image unchanged.
        """

        if image is None:
            raise ValueError("The 'image' input must be connected.")

        # ── 1. Verify input shape ───────────────────────────────

        if image.ndim != 4:                     # expected (B, H, W, C)
            raise ValueError(f"Image tensor must have shape (B,H,W,C), got {image.shape}")

        B, oh, ow, C = image.shape

        # If max_side == 0 – no resizing needed
        if max_side == 0:
            return (image,)

        # ── 2. Compute scaling factor based on the longest side ────────

        current_max = max(oh, ow)
        ratio = max_side / current_max           # >1 → enlarge, <1 → shrink

        new_w = max(1, round(ow * ratio))
        new_h = max(1, round(oh * ratio))

        # ── 3. Resize using PyTorch interpolation ───────────────────────

        img = image.permute(0, 3, 1, 2)           # (B, C, H, W)

        kwargs = {"size": (new_h, new_w)}
        if interpolation in ("linear", "bilinear", "bicubic", "trilinear"):
            kwargs["align_corners"] = False

        img_resized = F.interpolate(img, mode=interpolation, **kwargs)

        image_out = img_resized.permute(0, 2, 3, 1)   # back to (B, H, W, C)
        return (image_out,)

class MaskDebugNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "debug"

    def debug(self, mask):
        import torch
        t = mask.squeeze(-1) if mask.ndim == 4 and mask.shape[3] == 1 else mask
        return (f"shape={tuple(t.shape)}",)

class ShiftSliderNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shift": ("FLOAT", {"default": 0.0,
                                    "min": 0.0,
                                    "max": 100.0,
                                    "step": 0.01})
            }
        }

    RETURN_TYPES = ("FLOAT",)        
    RETURN_NAMES = ("shift",)        

    FUNCTION = "run"               
    CATEGORY = "utils"      

    def run(self, shift):
       
        return (shift,)                

# DA_Base_KSampler and DA_Enhanced_KSampler based on WAS_KSampler from https://github.com/WASasquatch/was-node-suite-comfyui (archived)

# By WASasquatch (Discord: WAS#0263)
#
# Copyright 2023 Jordan Thompson (WASasquatch)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
class DA_Base_KSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", ),
                "seed": ("INT", {"default": 0, "min": 0,"max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0,"max": 1.0, "step": 0.01}),
                }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

class DA_Enhanced_KSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0,"max": 100.0, "step":0.01}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Sampling"

    # ----------Helper method----------
    def _apply_shift(self, model: "MODEL", shift: float, multiplier: float = 1.0):
        m = model.clone()
        import comfy.model_sampling
        sampling_base   = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type   = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type): pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=multiplier)

        m.add_object_patch("model_sampling", model_sampling)
        return m
    # -------------------------------------------

    def sample(self,
               model: "MODEL",
               seed: int,
               steps: int,
               cfg: float,
               sampler_name: str,
               scheduler: str,
               positive: "CONDITIONING",
               negative: "CONDITIONING",
               latent_image: "LATENT",
               denoise: float = 1.0,
               shift: float = 0.0):
        if shift:
            try:
                model = self._apply_shift(model, shift)
            except Exception as e:
                print(f"[DA_Enhanced_KSampler] error applying Model_Shift: {e}")

        return nodes.common_ksampler(
            model,
            seed, steps, cfg,
            sampler_name, scheduler,
            positive, negative,
            latent_image,
            denoise=denoise
        )

#This node is based on CImageLoadWithMetadata from https://github.com/crystian/ComfyUI-Crystools
class LoadImageWithMetadataNode:
    """Loads an image from the input folder and returns a tensor,
       mask (if there is an alpha channel) and two types of metadata: full RAW and 'clean' – only parameters."""
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()

        exclude_files  = {"Thumbs.db", "*.DS_Store", "desktop.ini", "*.lock"}
        exclude_folders = {"clipspace", ".*"}

        file_list = []

        for root, dirs, files in os.walk(input_dir, followlinks=True):
            dirs[:]   = [d for d in dirs if not any(fnmatch.fnmatch(d, e) for e in exclude_folders)]
            files     = [f for f in files if not any(fnmatch.fnmatch(f, e) for e in exclude_files)]

            for file in files:
                relpath = os.path.relpath(os.path.join(root, file), start=input_dir)
                relpath = relpath.replace("\\", "/")          # windows patch
                file_list.append(relpath)

        return {"required": {"image": (sorted(file_list), {"image_upload": True})}}

    CATEGORY = "Image"

    # 4 outputs: image, mask, full RAW metadata, and 'clean' set only with parameters
    RETURN_TYPES = ("IMAGE", "MASK", "METADATA_RAW", "METADATA_CLEAN")
    RETURN_NAMES = ("image", "mask", "Metadata RAW", "Metadata Clean")

    OUTPUT_NODE = True
    FUNCTION   = "execute"

    # ---------------------------------------------------------------
    def execute(self, image):
        """Main function. Returns the image tensor,
           mask (if an alpha channel exists), full metadata and 'clean' parameters."""
        image_path = folder_paths.get_annotated_filepath(image)

        # build_metadata returns img, prompt (not used) and metadata
        img, _, metadata = build_metadata(image_path)

        # 1️. EXIF orientation handling
        img = ImageOps.exif_transpose(img)

        # 2️. Convert to tensor
        image_tensor = torch.from_numpy(
            np.array(img.convert("RGB")).astype(np.float32) / 255.0
        )[None,]                     # [1,H,W,C]

        # 3️. Mask (if there is an alpha channel)
        if 'A' in img.getbands():
            mask_np    = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask_torch = 1. - torch.from_numpy(mask_np)
        else:
            mask_torch = torch.zeros((64, 64), dtype=torch.float32)

        # 4️.'Clean' metadata – only parameters (if they exist)
        clean_meta = {}
        if "parameters" in metadata:
            clean_meta["parameters"] = metadata["parameters"]

        # --- Return
        return image_tensor, mask_torch.unsqueeze(0), metadata, clean_meta

    # ---------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, image):
        """Check for changes to the image (SHA‑256)."""
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.hexdigest()

    # ---------------------------------------------------------------
    @classmethod
    def VALIDATE_INPUTS(cls, image):
        """Check that the file exists."""
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

#feedbackNode, MyXYZHelper, MyXYGridAccumulator,  MyXYZSuperStacker nodes based on nodes from  https://github.com/kenjiqq/qq-nodes-comfyui
# --- BASE CLASS FOR PREVIEW ---
class FeedbackNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    def preview_images(self, images, filename_prefix="MyXYZ"):
        if not images or len(images) == 0:
            return []
        
        # Get save parameters from the first image
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        
        results = list()
        for image in images:
            # Convert tensor to array for PIL
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        return results

# --- 1. HELPER ---
class MyXYZHelper:
    _last_index = -1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "row_list": ("LIST",),
                "column_list": ("LIST",),
                "page_list": ("LIST",),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "row_prefix": ("STRING", {"default": ""}),
                "column_prefix": ("STRING", {"default": ""}),
                "page_prefix": ("STRING", {"default": ""}),
                "font_size": ("INT", {"default": 50}),
                "grid_gap": ("INT", {"default": 20}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "XYZ_GRID_CONTROL")
    RETURN_NAMES = ("row_value", "column_value", "page_value", "XYZ_GRID_CONTROL")
    FUNCTION = "run"
    CATEGORY = "Utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def run(self, row_list, column_list, page_list, index, **kwargs):
        force_reset = (index == 0) or (index < self._last_index)
        self._last_index = index

        len_x, len_y, len_z = len(column_list), len(row_list), len(page_list)
        total_per_page = len_x * len_y
        
        z_idx = (index // total_per_page) % len_z
        adj_idx = index % total_per_page
        row_idx = (adj_idx // len_x) % len_y
        col_idx = adj_idx % len_x

        r_pre = kwargs.get('row_prefix', "")
        c_pre = kwargs.get('column_prefix', "")
        p_pre = kwargs.get('page_prefix', "")

        row_ann = ";".join([f"{r_pre}: {str(x)}" if r_pre else str(x) for x in row_list])
        col_ann = ";".join([f"{c_pre}: {str(y)}" if c_pre else str(y) for y in column_list])
        page_label = f"{p_pre}: {str(page_list[z_idx])}" if p_pre else str(page_list[z_idx])

        XYZ_GRID_CONTROL = (
            total_per_page, 
            0 if force_reset else adj_idx, 
            row_ann, 
            col_ann, 
            len_x, 
            kwargs.get('font_size', 50), 
            kwargs.get('grid_gap', 20),
            page_label,
            z_idx,
            len_z,
            0 if force_reset else index
        )

        return (row_list[row_idx], column_list[col_idx], page_list[z_idx], XYZ_GRID_CONTROL)

# --- 2. ACCUMULATOR---
class MyXYGridAccumulator(FeedbackNode):
    image_batch = torch.Tensor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "XYZ_GRID_CONTROL": ("XYZ_GRID_CONTROL",),
                "show_previews": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Utils"

    def run(self, images, XYZ_GRID_CONTROL, show_previews, unique_id):
        count, reset_val, row_txt, col_txt, x_size, f_size, gap, z_label, *_ = XYZ_GRID_CONTROL
        
        if reset_val == 0:
            MyXYGridAccumulator.image_batch = images
        else:
            if MyXYGridAccumulator.image_batch.numel() == 0:
                MyXYGridAccumulator.image_batch = images
            else:
                MyXYGridAccumulator.image_batch = torch.cat((
                    MyXYGridAccumulator.image_batch, 
                    images.to(MyXYGridAccumulator.image_batch.device)
                ), dim=0)

        curr_num = MyXYGridAccumulator.image_batch.shape[0]

        if curr_num < count:
            ui_res = []
            if show_previews:
                preview_list = [MyXYGridAccumulator.image_batch[i] for i in range(curr_num)]
                ui_res = self.preview_images(preview_list)
            return {"result": (ExecutionBlocker(None),), "ui": {"images": ui_res}}
        
        else:
            page_imgs = MyXYGridAccumulator.image_batch[:count]
            MyXYGridAccumulator.image_batch = torch.Tensor()

            ui_res = []
            if show_previews:
                ui_res = self.preview_images([page_imgs[i] for i in range(count)])

            graph = GraphBuilder()
            ann = graph.node("GridAnnotation", row_texts=row_txt, column_texts=col_txt, font_size=f_size)
            grid = graph.node("ImagesGridByColumns", images=page_imgs, annotation=ann.out(0), max_columns=x_size, gap=gap)
            p_ann = graph.node("GridAnnotation", row_texts=" ", column_texts=z_label, font_size=int(f_size*1.5))
            final = graph.node("ImagesGridByColumns", images=grid.out(0), annotation=p_ann.out(0), max_columns=1, gap=0)
            
            return {"result": (final.out(0),), "ui": {"images": ui_res}, "expand": graph.finalize()}

# --- 3. SUPER STACKER (final batch) ---
class MyXYZSuperStacker:
    storage = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "XYZ_GRID_CONTROL": ("XYZ_GRID_CONTROL",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "stack"
    CATEGORY = "Utils"

    def stack(self, image, XYZ_GRID_CONTROL):
        *_, z_idx, total_z, g_index = XYZ_GRID_CONTROL
        
        if g_index == 0:
            MyXYZSuperStacker.storage = []

        if len(MyXYZSuperStacker.storage) == z_idx:
            MyXYZSuperStacker.storage.append(image)

        if len(MyXYZSuperStacker.storage) >= total_z:
            all_pages = torch.cat(MyXYZSuperStacker.storage, dim=0)
            MyXYZSuperStacker.storage = [] 
            return (all_pages,)
        else:
            return (ExecutionBlocker(None),)

class ListCreaterNode:
    """
    Splits the input multiline text by the specified delimiter 
    and returns a list of strings.
    
    Supports escaped sequences, for example "\n" → newline.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Separator": ("STRING", {"default": ","}),
                "Text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "split_text"
    CATEGORY = "utility/text"

    def split_text(self, Text: str, Separator: str):
        # Logic for the "\n" separator
        if Separator == r"\n":
            sep = "\n"
        else:
            sep = Separator
        parts = Text.split(sep)
        cleaned = [p.strip().replace('\n', ' ') for p in parts]

        return (cleaned,)
        
class CountListNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_list": ("LIST",)
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "get_length"

    CATEGORY = "Utils"

    def get_length(self, input_list):
        length = len(input_list)
        return (length,)
        
class AnytoIntegerAdapterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"input_any": ("*",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "adapt"
    CATEGORY = "utils"

    def adapt(self, input_any):
        """
        Converts the input data to an integer. If conversion is impossible, returns None
        """
        try:
            return (int(input_any),)
        except (ValueError, TypeError):
            print(f"Cannot convert '{input_any}' to an integer.")
            return (None,)
            
class AnytoFloatAdapterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"input_any": ("*",)}}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "adapt"
    CATEGORY = "utils"

    def adapt(self, input_any):
        """
        Converts the input data to an integer. If conversion is impossible, returns None
        """
        try:
            return (float(input_any),)
        except (ValueError, TypeError):
            print(f"Cannot convert '{input_any}' to an integer.")
            return (None,)

class SamplerSelectorFromStringNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name_str": ("STRING", {"default": "euler"}),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler_name",)
    FUNCTION = "get_names"
    CATEGORY = "utils"

    def get_names(self, sampler_name_str):
        if sampler_name_str not in comfy.samplers.KSampler.SAMPLERS:
            print(f"Warning: Sampler '{sampler_name_str}' not found. Fallback to euler.")
            return ("euler",)
        return (sampler_name_str,)

class SchedulerSelectorFromStringNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler_str": ("STRING", {"default": "normal"}),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "get_names"
    CATEGORY = "utils"

    def get_names(self, scheduler_str):
        if scheduler_str not in comfy.samplers.KSampler.SCHEDULERS:
            return ("normal",)
        return (scheduler_str,)
        
class ListRerouteNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_list": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "reroute"

    CATEGORY = "utils"

    def reroute(self, input_list):
        return (input_list,)

class StringToAnyNode:
    """
    Universal Bridge: Takes a string and returns it as type '*'
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any_output",)
    FUNCTION = "convert"
    CATEGORY = "utils"

    def convert(self, input_string):
        return (input_string,)

class XYZConflictValidatorAndSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_type": (["STRING", "INT", "FLOAT", "*"],),
                "global_val": ("*",), 
            },
            "optional": {
                "row": ("*",),
                "column": ("*",),
                "pages": ("*",),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "utils/XYZ"

    def execute(self, output_type, global_val, **kwargs):
        # 1. Multiple choice check (Axis conflict)
        active_inputs = {k: v for k, v in kwargs.items() if v is not None}
        
        if len(active_inputs) > 1:
            raise ValueError(f"⚠️ XYZ Conflict: Few active parameters of the same type: {list(active_inputs.keys())}")

        # Choice value (XYZ или Global)
        raw_result = active_inputs[next(iter(active_inputs))] if active_inputs else global_val

        # 2. Type casting according to the selection in the menu
        try:
            if output_type == "INT":
                return (int(float(raw_result)),)
            elif output_type == "FLOAT":
                return (float(raw_result),)
            elif output_type == "STRING":
                return (str(raw_result),)
            else:
                return (raw_result,)
        except (ValueError, TypeError):
            raise TypeError(f"⚠️ XYZ Type Mismatch: Unable to turn '{raw_result}' into a {output_type}")

class ListCombinerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_a": ("LIST",),
                "list_b": ("LIST",) 
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "join_lists"
    CATEGORY = "List Operations" 

    def join_lists(self, list_a, list_b):
        return (list_a + list_b,)

class BooleanSwitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {        
                "condition": ("BOOLEAN",)
            },
            "optional": {          
                "on_true": ("*",), 
                "on_false": ("*",)
            }
        }

    RETURN_TYPES = ("*",)    
    FUNCTION = "switch"      
    CATEGORY = "logic"    

    def switch(self, condition: bool, on_true=None, on_false=None):
        if condition:
            return (on_true if on_true is not None else None,)
        else:
            return (on_false if on_false is not None else None,)

class SaveImageNoMetaNode:
    """
    Saves an image without workflow/metadata.
    Supports %date% placeholder which is replaced by yyyy-mm-dd.
    Always saves the file inside the `output/<relative path>` folder.
    An index is added automatically (00001, 00002 ...).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "path": ("STRING",
                         {"default": "ComfyUI",
                          "tooltip": "Relative path inside `output`. Use %date% for yyyy-mm-dd."}),
                "preview": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "ImageSaver"
    OUTPUT_NODE = True

    def _ensure_rgb_uint8(self, img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # If a batch is received (N, H, W, C), take the first frame
        if img.ndim == 4:
            img = img[0]
        while img.ndim > 3:
            img = img[0]
        if img.ndim == 2:                # H,W → RGB
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:         # H,W,1 → RGB
            img = np.concatenate([img] * 3, axis=-1)
        if img.dtype.kind in "fc":       # float → 0-255
            img = np.clip(img * 255.0, 0, 255)
        else:
            img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def _unique_name(self, target_path: Path) -> Path:
        """Returns the path with an incremented index if the file already exists."""
        if not target_path.exists():
            return target_path
        stem = target_path.stem
        suffix = target_path.suffix if target_path.suffix else ".png"
        counter = 1
        while True:
            new_name = f"{stem}_{counter:05d}{suffix}"
            candidate = target_path.parent / new_name
            if not candidate.exists():
                return candidate
            counter += 1

    def save(self, image, path: str, preview: bool):
        if not path:
            raise ValueError("Save path is not specified")

       # Replace %date% with the current date in the format yyyy-mm-dd
        current_date = datetime.now().strftime("%Y-%m-%d")
        processed_path = path.replace("%date%", current_date)
        # ------------------------------------
            
        # 1. Define the root of the output folder
        # In ComfyUI, it's better to use the standard output directory logic
        # But staying with your implementation:
        root_dir = Path(os.getcwd())
        output_base = root_dir/ "output"
        # Correction: using your original variable name
        output_base = Path(os.getcwd()) / "output"
        
        # 2. Formulate the target file path inside 'output'
        clean_relative_path = processed_path.lstrip("/\\").lstrip("./")
        target_file_path = output_base / clean_relative_path
        
        # If no extension is provided, add .png
        if target_file_path.suffix == "":
            target_file_path = target_file_path.with_suffix(".png")
            
        # 3. Generate a unique name (with index)
        unique_path = self._unique_name(target_file_path)
        
        # 4. Save the image
        img_np = self._ensure_rgb_uint8(image)
        pil_img = Image.fromarray(img_np)
        pil_img.info.clear()  # Remove all metadata (workflow, etc.)
        unique_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(str(unique_path))
        
        # 5. Formulate the response for UI (preview)
        if preview:
            try:
                subfolder = str(unique_path.parent.relative_to(output_base))
            except ValueError:
                subfolder = ""
                
            if subfolder in [".", ""]:
                subfolder = ""
            else:
                subfolder = subfolder.replace(os.sep, '/')
                
            return {
                "ui": {
                    "images": [
                        {
                            "filename": unique_path.name,
                            "subfolder": subfolder,
                            "type": "output"
                        }
                    ]
                }
            }
        return {}            