import comfy.samplers
import folder_paths 
import os
import torch
import torch.nn.functional as F
import comfy
from pathlib import Path
import numpy as np
from PIL import Image
import nodes
import comfy.samplers

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
    List of all LORA files from the `loras/` folder.
    The full path will be used as the value in the dropdown menu.
    """
    loras = folder_paths.get_filename_list("loras")      # <-- full path
    return ["none"] + loras
    
class SamplerGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        samplers = get_sampler_list()
        inputs = {"required": {f"sampler_{i+1}": (samplers, {"default": "none"}) for i in range(10)}}
        return inputs
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"sampler_{i+1}")
            if name and name != "none": selected.append(name)
        return (", ".join(selected),)

class SchedulerGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        schedulers = get_scheduler_list()
        inputs = {"required": {f"scheduler_{i+1}": (schedulers, {"default": "none"}) for i in range(10)}}
        return inputs
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"scheduler_{i+1}")
            if name and name != "none": selected.append(name)
        return (", ".join(selected),)

class ModelGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list()
        inputs = {"required": {f"model_{i+1}": (models, {"default": "none"}) for i in range(10)}}
        return inputs
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("MODEL_STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"model_{i+1}")
            if name and name != "none": selected.append(name)
        return (", ".join(selected),)

class DiffusionModelGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_diffusion_model_file_list()
        inputs = {"required": {f"diff_model_{i+1}": (models, {"default": "none"}) for i in range(10)}} 
        return inputs
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DIFF_MODEL_STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"diff_model_{i+1}")
            if name and name != "none": selected.append(name)
        return (", ".join(selected),)

class AnyAdapterNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Switched from "required" to "optional" – this node can accept any type.
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

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"

    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"vae_{i+1}")
            if name and name != "none":
                selected.append(name)
        return (", ".join(selected),)  

class TextEncoderGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        encoders = get_text_encoder_list()
        inputs = {
            "required": {f"text_enc_{i+1}": (encoders, {"default": "none"}) for i in range(10)}
        }
        return inputs

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"

    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"text_enc_{i+1}")
            if name and name != "none":
                selected.append(name)
        return (", ".join(selected),)

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
            # Dropdown for the full path of a LoRA file
            inputs["required"][f"lora_{i+1}"] = (loras, {"default": "none"})
            # Corresponding weight input
            inputs["required"][f"weight_{i+1}"] = (
                "FLOAT",
                {"default": 1.0, "min": 0.00, "max": 10.00, "step": 0.01},
            )
        return inputs

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"

    def generate_string(self, **kwargs):
        parts = []
        for i in range(10):
            # Full path from the dropdown
            path = kwargs.get(f"lora_{i+1}")
            weight = kwargs.get(f"weight_{i+1}")

            if not path or path == "none":
                continue

            # Take only the file name without extension
            name_without_ext = os.path.splitext(os.path.basename(path))[0]
            weight_str = f"{float(weight):.2f}"
            parts.append(f"<lora:{name_without_ext}:{weight_str}>")

        return (", ".join(parts),)

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
        stays untouched.  Returns None for the output that was not given.
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
            image_out = None

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
            mask_out = None

        return image_out, mask_out, new_width, new_height

class ResizeMethodControlNode:
    """
    Outputs the chosen resize method.
    Can be connected to the 'method' input of ImageResizeNode.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Same enumeration as in ImageResizeNode
                "method": ([
                    "stretch",
                    "keep proportion",
                    "fill / crop",
                    "pad"
                ],),
            }
        }

    RETURN_TYPES = ("STRING",)          # one output element – string
    FUNCTION    = "set"                 # name of the method to be called
    CATEGORY    = "utils"

    def set(self, method: str):
        """Return the selected method as a single output."""
        return (method,)

class ResizeInterpolationControlNode:
    """
    Outputs the chosen interpolation type.
    Can be connected to the 'interpolation' input of ImageResizeNode.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # List of all options from ImageResizeNode
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

    RETURN_TYPES = ("STRING",)          # one output element – string
    FUNCTION    = "set"                 # name of the method to be called
    CATEGORY    = "utils"

    def set(self, interpolation: str):
        """Return the selected interpolation type as a single output."""
        return (interpolation,)

class AnyConcatNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            # Delimiter: always a text field, can be changed manually
            "required": {"delimiter": ("STRING", {"default": " "})},

            # text1…text5 are only connectors. Their type "*" means “any value”, but in the UI they appear as empty slots without a text field.
            "optional": {f"text{i}": ("*",) for i in range(1, 6)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat"
    CATEGORY = "utils"

    def concat(self, delimiter: str, **kwargs):
        """
        kwargs contains only those slots that were actually connected.
        If a slot was not connected, it simply is absent from the dict.
        """
        # Convert everything to string (so you can concatenate numbers and other types) and remove empty/None values.
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