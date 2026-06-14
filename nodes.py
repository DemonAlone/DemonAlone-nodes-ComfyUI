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
import latent_preview
import pickle
import logging
logger = logging.getLogger(__name__)
import math
import comfy.model_management
import comfy.sample
import comfy.utils
import sys

try:
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
    UPSCALE_AVAILABLE = True
except ImportError:
    UPSCALE_AVAILABLE = False
    print("[TiledESRGANUpscaler] Warning: comfy_extras.nodes_upscale_model not found. Upscale model will be ignored.")

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
            logger.warning("piexif error on WebP – ignore")

    return img, prompt, metadata
    
def create_grid_annotation(
    image: Image.Image,
    column_texts: list = None,
    row_texts: list = None,
    font: ImageFont.FreeTypeFont = None,
    text_color: tuple = (255, 255, 255),
    bg_color: tuple = (0, 0, 0),
    
    ) -> Image.Image:
    """
    Adds annotations (text above and left) to the image.
    Returns a new image with fields for annotations.
    """
    # If no column or row texts provided, return original image
    if column_texts is None and row_texts is None:
        return image

    # Try to use a specific font if not provided
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

    # Calculate the height needed for text in columns
    draw = ImageDraw.Draw(image)
    col_height = 0
    if column_texts:
    # Calculate the height needed for text in columns
        for txt in column_texts:
            bbox = draw.textbbox((0, 0), txt, font=font)
            col_height = max(col_height, bbox[3] - bbox[1])
        col_height += 10  # Add small spacing between text

    # Calculate the width needed for rows
    row_width = 0
    if row_texts:
        for txt in row_texts:
            bbox = draw.textbbox((0, 0), txt, font=font)
            row_width = max(row_width, bbox[2] - bbox[0])
        row_width += 10
    # Get original image dimensions
    orig_w, orig_h = image.size
    # Get original image dimensions

    new_w = orig_w + row_width
    new_h = orig_h + col_height

    # Create a new blank image with background color
    new_img = Image.new("RGB", (new_w, new_h), bg_color)
    # Create a new blank image with background color
    new_img.paste(image, (row_width, col_height))

    draw = ImageDraw.Draw(new_img)

    # Create a new blank image with background color
    if column_texts:
        # Create a new blank image with background color
        col_w = orig_w // len(column_texts)
        for i, txt in enumerate(column_texts):
            bbox = draw.textbbox((0, 0), txt, font=font)
            text_w = bbox[2] - bbox[0]
            x = row_width + i * col_w + (col_w - text_w) // 2
            y = (col_height - (bbox[3] - bbox[1])) // 2
            draw.text((x, y), txt, fill=text_color, font=font)

    # Create a new blank image with background color
    if row_texts:
        row_h = orig_h // len(row_texts)
        for i, txt in enumerate(row_texts):
            bbox = draw.textbbox((0, 0), txt, font=font)
            text_h = bbox[3] - bbox[1]
            y = col_height + i * row_h + (row_h - text_h) // 2
            x = (row_width - (bbox[2] - bbox[0])) // 2
            draw.text((x, y), txt, fill=text_color, font=font)

    return new_img

def images_grid_by_columns(
    images: list,
    max_columns: int,
    gap: int = 0,
    bg_color: tuple = (0, 0, 0),
    ) -> Image.Image:
    """
    Creates a grid layout from a list of PIL images.
    max_columns - maximum number of columns in the grid
    gap - spacing between columns
    bg_color - background color for the grid
    """
    if not images:
        raise ValueError("List of images is empty")

    n = len(images)
    columns = min(max_columns, n) # Limit to maximum allowed columns
    rows = (n + columns - 1) // columns # Limit to maximum allowed columns

    # Limit to maximum allowed columns
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)

    total_w = columns * cell_w + (columns - 1) * gap
    total_h = rows * cell_h + (rows - 1) * gap
    # Limit to maximum allowed columns
    grid = Image.new("RGB", (total_w, total_h), bg_color)
    
    # Place each image into its appropriate position in the grid
    for idx, img in enumerate(images):
        row = idx // columns
        col = idx % columns
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)
        # Center the image within its cell
        offset_x = (cell_w - img.width) // 2
        offset_y = (cell_h - img.height) // 2
        grid.paste(img, (x + offset_x, y + offset_y))

    return grid
    
class SamplerGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        samplers = get_sampler_list()
        inputs = {"required": {f"sampler_{i+1}": (samplers, {"default": "none"}) for i in range(10)}}
        return inputs
    RETURN_TYPES = ("STRING", "LIST")  
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    DESCRIPTION = "Generates a comma-separated string and list of any selected samplers from up to 10 inputs. Useful for dynamically constructing sampler lists based on user selection before feeding them into sampling nodes."
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
    DESCRIPTION = "Generates a comma-separated string and list of any selected schedulers from up to 10 inputs. Used to dynamically assemble scheduler configurations from multiple source nodes before applying them to the sampling process."
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
    DESCRIPTION = "Aggregates standard diffusion model weights from up to 10 inputs. Generates a combined list and comma-separated string of selected checkpoint files for flexible pipeline configuration."
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
    DESCRIPTION = "Collects custom or pre-processed diffusion model file paths from multiple sources (up to 10 inputs). Outputs a consolidated list and string representation for seamless integration into multi-model workflows."
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
    DESCRIPTION = "A flexible pass-through node that accepts any input type. It safely forwards data downstream or returns a clean None output if the input is unconnected, ensuring pipeline stability without breaking custom workflows."
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
    CATEGORY = "utils"
    DESCRIPTION = "Retrieves checkpoint model names from the ComfyUI checkpoints folder and outputs both the filename tuple (for combo menus) and its string representation, allowing dynamic model selection in utility chains."
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
    DESCRIPTION = "Aggregates up to 10 selected VAE models into a single comma-separated string and a corresponding list. Designed to dynamically build a chain of VAEs for processing workflows that require multiple decode/encode stages."
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
    DESCRIPTION = "Consolidates up to 10 selected text encoders (CLIP models) into a unified string output and list. Enables the dynamic assembly of multi-stage conditioning pipelines by combining various encoder variants."

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
    CATEGORY = "utils"
    DESCRIPTION = "Selects a single VAE model from the available list. Outputs both the model path (for direct connection to nodes like Load VAE) and a plain text string representation of the selection for use in other utility nodes."
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
    DESCRIPTION = "Selects a single Text Encoder (CLIP model) from the installed encoders. Provides both the encoder path compatible with loading nodes and a corresponding string output, enabling flexible routing of conditioning models within the workflow."
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
    DESCRIPTION = "Safely converts a text input string into an integer. Automatically handles parsing errors by logging them and returning 0 as a fallback, ensuring the workflow continues without crashing on invalid numeric strings."
    
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
    DESCRIPTION = "Converts a text input string into a floating-point number with error handling. If the string cannot be parsed as a valid float, it logs the issue and returns 0.0 to keep the pipeline running smoothly."

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
    DESCRIPTION = "Merges up to 5 optional text inputs into a single continuous string, using a configurable delimiter."

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
    DESCRIPTION = "Aggregates up to 10 LoRA models with individually assigned weights, generating both a formatted prompt string and a structured list. Supports dynamic inclusion of empty values (placeholders) alongside active LoRAs for flexible pipeline construction."

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

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)        # optional – gives the output a name
    FUNCTION = "get_value"           # method that will be invoked
    CATEGORY = "utils"
    DESCRIPTION = "Outputs an integer clip skip value ranging from -24 to -1. Provides fine-grained control over the depth of CLIP token skipping in diffusion models."

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
    DESCRIPTION = "Dynamically constructs a prompt prefix string containing Pony-specific score, rating, and source tags (e.g., score_9, rating_safe). Combines selected variants into a single comma-separated string for immediate use in text encoding."
    
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
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "width": ("INT", {"default": 512, "min": 0, "max": 16834}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16834}),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],),
                "multiple_of": ("INT", {"default": 1, "min": 1, "max": 512, "description": "1 = disable, otherwise round down to multiple"})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image_out", "mask_out", "width", "height")
    FUNCTION = "execute"
    DESCRIPTION = "A versatile utility node for synchronously resizing images and masks. Supports multiple scaling methods (stretch, keep proportion, fill/crop, pad), various interpolation filters, and conditional logic to resize only when necessary. Automatically handles empty outputs if an input tensor is missing."
    CATEGORY = "utils"

    def execute(self, image=None, mask=None, width=512, height=512, method="stretch",
                interpolation="nearest", condition="always", multiple_of=64):
        has_image = image is not None
        has_mask = mask is not None
        if not (has_image or has_mask):
            raise ValueError("At least one of 'image' or 'mask' must be connected")

        # source dimensions
        source = image if has_image else mask
        if source.ndim == 4:
            _, oh, ow, _ = source.shape
        elif source.ndim == 3:
            _, oh, ow = source.shape
        else:
            raise ValueError(f"Unsupported shape: {source.shape}")

        # target dimensions (final)
        final_w = width if width > 0 else ow
        final_h = height if height > 0 else oh

        # intermediate variables
        resize_w, resize_h = final_w, final_h
        pad_l = pad_r = pad_t = pad_b = 0
        crop_x = crop_y = crop_x2 = crop_y2 = None

        if method == "stretch":
            resize_w, resize_h = final_w, final_h

        elif method == "keep proportion":
            ratio = min(final_w / ow, final_h / oh)
            resize_w = round(ow * ratio)
            resize_h = round(oh * ratio)

        elif method == "pad":
            ratio = min(final_w / ow, final_h / oh)
            resize_w = round(ow * ratio)
            resize_h = round(oh * ratio)
            pad_l = (final_w - resize_w) // 2
            pad_r = final_w - resize_w - pad_l
            pad_t = (final_h - resize_h) // 2
            pad_b = final_h - resize_h - pad_t

        elif method == "fill / crop":
            ratio = max(final_w / ow, final_h / oh)
            resize_w = round(ow * ratio)
            resize_h = round(oh * ratio)
            crop_x = (resize_w - final_w) // 2
            crop_y = (resize_h - final_h) // 2
            crop_x2 = crop_x + final_w
            crop_y2 = crop_y + final_h
            if crop_x2 > resize_w:
                crop_x -= (crop_x2 - resize_w)
            if crop_x < 0:
                crop_x = 0
            if crop_y2 > resize_h:
                crop_y -= (crop_y2 - resize_h)
            if crop_y < 0:
                crop_y = 0
            crop_x2 = crop_x + final_w
            crop_y2 = crop_y + final_h

        # condition for resizing
        should_resize = (
            condition == "always" or
            (condition == "downscale if bigger" and (oh > final_h or ow > final_w)) or
            (condition == "upscale if smaller" and (oh < final_h or ow < final_w)) or
            (condition == "if bigger area" and (oh * ow > final_h * final_w)) or
            (condition == "if smaller area" and (oh * ow < final_h * final_w))
        )

        # --- image processing ---
        if has_image:
            img = image.permute(0, 3, 1, 2)  # B,C,H,W
            if should_resize:
                if interpolation == "lanczos" and comfy is not None:
                    img = comfy.utils.lanczos(img, resize_w, resize_h)
                else:
                    kwargs = {"size": (resize_h, resize_w)}
                    if interpolation in ("linear", "bilinear", "bicubic", "trilinear"):
                        kwargs["align_corners"] = False
                    img = F.interpolate(img, mode=interpolation, **kwargs)

                if method == "pad":
                    img = F.pad(img, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                elif method == "fill / crop":
                    img = img[:, :, crop_y:crop_y2, crop_x:crop_x2]

            image_out = img.permute(0, 2, 3, 1)
        else:
            batch = source.shape[0]
            image_out = torch.zeros(batch, final_h, final_w, 3)

        # --- mask processing ---
        if has_mask:
            if mask.ndim == 3:
                msk = mask.unsqueeze(1)
            elif mask.ndim == 4 and mask.shape[3] == 1:
                msk = mask.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")

            if should_resize:
                msk = F.interpolate(msk, size=(resize_h, resize_w), mode='nearest')
                if method == "pad":
                    msk = F.pad(msk, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                elif method == "fill / crop":
                    msk = msk[:, :, crop_y:crop_y2, crop_x:crop_x2]

            mask_out = msk.squeeze(1)
        else:
            batch = source.shape[0]
            mask_out = torch.zeros(batch, final_h, final_w)

        # --- apply multiple_of (If >1) ---
        if multiple_of > 1:
            final_w = (final_w // multiple_of) * multiple_of
            final_h = (final_h // multiple_of) * multiple_of
            # important: crop/pad image and mask to new dimensions
            if final_w != image_out.shape[2] or final_h != image_out.shape[1]:
                # Simple centered cropping
                image_out = image_out[:, :final_h, :final_w, :]
                mask_out = mask_out[:, :final_h, :final_w]

        return image_out, mask_out, final_w, final_h


class ResizeMethodControlNode:
    """
    Remote control unit with resizing method.
    Sends the selected value as a combo type for compatibility.
    Should be connected to the 'method' input of ImageResizeNode.
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
    DESCRIPTION = "An external controller panel specifically for setting the resizing strategy of ImageResizeNode. Outputs the selected method (stretch, keep proportion, fill/crop, or pad) as a combo parameter to be connected directly to the main node's 'method' input."

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
    DESCRIPTION = "An external controller panel for defining the resampling filter used by ImageResizeNode. Allows remote configuration of interpolation types (nearest, bilinear, bicubic, area, lanczos, etc.) by outputting the selected value to the main node's 'interpolation' input."

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
    DESCRIPTION = "Concatenates any number of up to 5 text inputs into a single string using a custom delimiter. Acts as a flexible joiner that automatically ignores unconnected slots, ideal for building dynamic text prompts or combining parameters from various sources."

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

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION     = "merge"
    CATEGORY     = "conditioning"    
    DESCRIPTION = "Merges 1 to 3 conditioning inputs (text embeddings) into a single output by averaging their weights and layering them element-wise. Automatically calculates the blend weight based on the number of active connections, ensuring seamless integration of multiple conditioning signals without manual scaling."

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
    DESCRIPTION  = "Resizes an image to fit within a specified maximum dimension while strictly preserving its aspect ratio. Supports multiple interpolation modes (e.g., bilinear, bicubic, lanczos)"
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
    DESCRIPTION = "Inspects and reports the tensor shape of a connected mask node in string format. Useful for debugging pipeline issues related to dimension mismatches or verifying input consistency before further processing steps."

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
    DESCRIPTION = "Applies a dynamic shift value to the sampling process or subsequent operations."
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
    DESCRIPTION = "Executes standard diffusion sampling using the vanilla KSampler implementation"

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
                "preview_freq": ("INT", {"default": 1, "min": 1, "max": 100, "tooltip": "How often to update the preview (e.g., 2 for every other step)."})
            }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Sampling"
    DESCRIPTION = "An advanced KSampler supporting Model Shift (SDE), dynamic noise masking, and customizable preview frequency. Allows users to control how often latent previews are rendered during the sampling process to optimize memory usage and visual feedback."
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
               shift: float = 0.0,
               preview_freq: int = 1):
                    
        if shift:
            try:
                model = self._apply_shift(model, shift)
            except Exception as e:
                print(f"[DA_Enhanced_KSampler] error applying Model_Shift: {e}")
        # 1. Data preparation (almost identical to common_ksampler)
        device = comfy.model_management.get_torch_device()
        latent_samples = latent_image["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)
        
        # Noise preparation
        batch_inds = latent_image.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        
        noise_mask = latent_image.get("noise_mask", None)
        
        # 2. Create a custom callback function for preview rendering
        #    The logic for "preview every Nth step" is implemented here
        preview_format = "JPEG"
        previewer = latent_preview.get_previewer(device, model.model.latent_format)
        pbar = comfy.utils.ProgressBar(steps)
        
        def custom_callback(step, x0, x, total_steps):
            # Render preview only if the current step (step) is a multiple of preview_freq
            # or on the final step
            if (step + 1) % preview_freq == 0 or (step + 1) == total_steps:
                preview_bytes = None
                if previewer:
                    preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
                # Update progress bar with new image
                pbar.update_absolute(step + 1, total_steps, preview_bytes)
            else:
                # Update progress bar only, without preview
                pbar.update_absolute(step + 1, total_steps, None)
        # 3. Run sampling directly, passing our custom callback
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_samples,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=noise_mask,
            callback=custom_callback,
            disable_pbar=False,
            seed=seed
        )
        
        # 4. Format output data
        out = latent_image.copy()
        out["samples"] = samples
        return (out,)

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
    DESCRIPTION = "Loads an image from the input directory, applies EXIF orientation correction, and outputs the image tensor along with an alpha mask (if present). Additionally, it provides both full raw metadata and a filtered 'clean' version containing only essential parameters for downstream workflows."
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
    DESCRIPTION = "Orchestrates the grid layout by mapping the current execution index to specific row, column, and page values. Dynamically generates annotations (e.g., 'Value: 5') for the XYZ plot headers based on input lists and styling parameters."

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
    DESCRIPTION = "Buffers individual images into a visual grid as the XYZ loop progresses. Handles the accumulation of images per page, renders the preview grid with axis labels when full, and clears the batch upon completion of each slice."

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
        # ---- If the page is not yet filled ----
        curr_num = MyXYGridAccumulator.image_batch.shape[0]

        if curr_num < count:
            ui_res = []
            if show_previews:
                preview_list = [MyXYGridAccumulator.image_batch[i] for i in range(curr_num)]
                ui_res = self.preview_images(preview_list)
            return {"result": (ExecutionBlocker(None),), "ui": {"images": ui_res}}
        # ---- Page is filled, constructing the grid ----
        page_imgs = MyXYGridAccumulator.image_batch[:count]
        MyXYGridAccumulator.image_batch = torch.Tensor()  # Clear for the next page

        # 1. Convert tensors to PIL.Image
        pil_images = []
        for i in range(count):
            img_tensor = page_imgs[i]
            # Tensor shape is [H, W, C] with values 0..1
            np_img = (255. * img_tensor.cpu().numpy()).astype(np.uint8)
            pil_images.append(Image.fromarray(np_img))

        # 2. Prepare font
        try:
            # Attempt to load a standard system font
            font = ImageFont.truetype("arial.ttf", f_size)
        except:
            font = ImageFont.load_default()

        # 3. Build the main image grid (rows × columns)
        grid_img = images_grid_by_columns(pil_images, max_columns=x_size, gap=gap)

        # 4. Add row and column annotations
        row_texts = row_txt.split(";") if row_txt else []
        column_texts = col_txt.split(";") if col_txt else []

        if row_texts or column_texts:
            grid_img = create_grid_annotation(
                grid_img,
                column_texts=column_texts,
                row_texts=row_texts,
                font=font,
            )

        # 5. Add the page header (z_label) as a separate line on top
        if z_label:
            # Create a temporary 1x1 grid containing only the label image
            # and add the annotation with z_label
            # To maintain proportions, we could just draw text above the existing grid,
            # but following the original pattern: create a 1x1 grid with an annotation-header
            try:
                font_big = ImageFont.truetype("arial.ttf", int(f_size * 1.5))
            except:
                font_big = ImageFont.load_default()
            grid_img = create_grid_annotation(
                grid_img,
                column_texts=[z_label],
                row_texts=[" "],  # Empty space on the left
                font=font_big,
            )

        # 6. Convert back to ComfyUI tensor format
        final_np = np.array(grid_img).astype(np.float32) / 255.0
        final_tensor = torch.from_numpy(final_np).unsqueeze(0)  # [1, H, W, C]

        # 7. Preview (if enabled)
        ui_res = []
        if show_previews:
            # Display individual images of the current page (not the final grid)
            ui_res = self.preview_images([page_imgs[i] for i in range(count)])

        return {"result": (final_tensor,), "ui": {"images": ui_res}}

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
    DESCRIPTION = "Collects all rendered XY pages into a single multi-page image sequence once the full XYZ dataset is generated. Acts as the final aggregation node to output the complete result set for saving or further processing."

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
    DESCRIPTION = "Splits multiline text strings into a list of individual items using a custom delimiter (supports commas, newlines, etc.). Useful for parsing batch configurations or breaking down complex prompts into separate components."

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
    DESCRIPTION = "Calculates and returns the total number of elements in an input list. Provides a quick way to determine batch sizes, list lengths, or count available inputs within your workflow logic."
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
    DESCRIPTION = "Safely converts any compatible input value into an integer. If the conversion fails or is impossible, it returns None without crashing the workflow."
    
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
    DESCRIPTION = "Safely converts any compatible input value into a floating-point number. If the conversion fails or is impossible, it returns None to prevent workflow errors."

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
    DESCRIPTION = "Converts a string input into a valid KSampler sampler object. Automatically validates the input and falls back to 'euler' if an unsupported sampler name is provided, ensuring workflow stability."

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
    DESCRIPTION = "Parses a string input to select a compatible KSampler scheduler. Includes basic validation to handle incorrect inputs gracefully by defaulting to the 'normal' scheduler when necessary."

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
    DESCRIPTION = "Passes a list input through unchanged. Acts as a connector or placeholder in the graph to manage node connections without altering data content."
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
    DESCRIPTION = "Converts a string value into a universal type-compatible output, allowing seamless bridging between different data types."

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
    DESCRIPTION = "Safeguards the XYZ pipeline against type mismatches by enforcing that only one active parameter exists per execution step. Automatically casts and outputs the current grid value (Row, Column, or Page) as an Int, Float, or String based on configuration."
    
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
    DESCRIPTION = "Merges two input lists into a single combined list, preserving the order of elements from both sources. Essential for chaining multiple generated lists (e.g., samplers or encoders) into one cohesive pipeline stage."
    
    def join_lists(self, list_a, list_b):
        return (list_a + list_b,)

class BooleanSwitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {        
                "state": ("BOOLEAN",)
            },
            "optional": {          
                "on_true": ("*",), 
                "on_false": ("*",)
            }
        }

    RETURN_TYPES = ("*",)    
    FUNCTION = "switch"      
    CATEGORY = "logic"    
    DESCRIPTION = "A conditional routing node that outputs either the 'true' or 'false' input value based on the current boolean state."

    def switch(self, state: bool, on_true=None, on_false=None):
        if state:
            return (on_true if on_true is not None else None,)
        else:
            return (on_false if on_false is not None else None,)

class SaveImageNoMetaNode:
    """
    Saves an image without workflow/metadata.
    Supports %date% placeholder which is replaced by yyyy-mm-dd.
    Supports png and jpg formats.
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
                "format": (["png", "jpg"],),
                "preview": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "ImageSaver"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves images to the output folder without embedding workflow metadata. Supports PNG/JPEG formats, automatic date stamping (%date%), and auto-indexing (e.g., 00001.png) for duplicate files. Always strips EXIF/metadata to ensure clean output files."

    def _ensure_rgb_uint8(self, img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # If a batch is received (N, H, W, C), take the first frame
        if img.ndim == 4:
            img = img[0]
        while img.ndim > 3:
            img = img[0]
            
        # Ensure 3 channels (RGB) - important for JPG which doesn't support Alpha
        if img.ndim == 2:                # H,W → RGB
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:         # H,W,1 → RGB
            img = np.concatenate([img] * 3, axis=-1)
        elif img.shape[-1] == 4:         # RGBA → RGB (JPG doesn't support transparency)
            img = img[:, :, :3]

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

    def save(self, image, path: str, format: str, preview: bool):
        if not path:
            raise ValueError("Save path is not specified")
            
        # Replace %date% with the current date in the format yyyy-mm-dd
        current_date = datetime.now().strftime("%Y-%m-%d")
        processed_path = path.replace("%date%", current_date)
        
        # 1. Define the root of the output folder
        output_base = Path(os.getcwd()) / "output"
        
        # 2. Formulate the target file path inside 'output'
        clean_relative_path = processed_path.lstrip("/\\").lstrip("./")
        target_file_path = output_base / clean_relative_path
        
        # Handle extension logic: replace existing extension with chosen format
        if target_file_path.suffix != "" and target_file_path.suffix[1:].lower() != format:
            target_file_path = target_file_path.with_suffix(f".{format}")
        elif target_file_path.suffix == "":
            target_file_path = target_file_path.with_suffix(f".{format}")

        # 3. Generate a unique name (with index)
        unique_path = self._unique_name(target_file_path)
            
        # 4. Save the image
        img_np = self._ensure_rgb_uint8(image)
        pil_img = Image.fromarray(img_np)
        
        # Remove all metadata (workflow, etc.)
        pil_img.info.clear()  
        
        unique_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with format specific settings
        if format.lower() in ["jpg", "jpeg"]:
            pil_img.save(str(unique_path), format="JPEG", quality=95)
        else:
            pil_img.save(str(unique_path), format="PNG")
        
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

class DA_BusInNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {f"input{i}": ("*",) for i in range(1, 11)}
        }
   
    RETURN_TYPES = ("BUS_PIPE",)
    RETURN_NAMES = ("bus_pipe",)
    
    FUNCTION = "execute"
    CATEGORY = "System/Bus"
    DESCRIPTION = "Aggregates up to 10 connections into a single bus pipe. Preserves positions of empty inputs."
    
    def execute(self, **kwargs):
        # Initializing a fixed list of 10 elements set to None
        # This guarantees that input1 is always at index 0, input2 at 1 and so on.
        bus_data = [None] * 10 
        
        for i in range(1, 11):
            key = f"input{i}"
            # Check for key existence. If it exists (even if value is None), 
            if key in kwargs:
                bus_data[i-1] = kwargs[key]
        
        return (tuple(bus_data),)

class DA_BusOutNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"bus_pipe": ("BUS_PIPE",)}
        }
    
    RETURN_TYPES = tuple(["*"] * 10)
    RETURN_NAMES = tuple([f"output{i+1}" for i in range(10)])

    OUTPUT_IS_LIST = (False,) * 10
    
    FUNCTION = "execute"
    CATEGORY = "System/Bus"
    DESCRIPTION = "Unpacks the bus pipe back into up to 10 outputs."
    
    def execute(self, bus_pipe):
        # Convert to tuple type for safety
        if not isinstance(bus_pipe, tuple):
            bus_pipe = tuple(bus_pipe)
        
        outputs = []
        for i in range(10):
            item = bus_pipe[i] if i < len(bus_pipe) else None
            outputs.append(item)
        return tuple(outputs)

class WanNumFramesNode:
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("num_frames",)
    DESCRIPTION  = "Output integer value with strict range constraints (min: 1, max: 10000, step: 4). " 
    FUNCTION     = "execute"
    CATEGORY     = "utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_frames": ("INT", {"default": 49, 
                                   "min": 1, 
                                   "max": 10000, 
                                   "step": 4}), 
            }
        }

    def execute(self, num_frames=None):
        """
        Returns the integer value provided by the input widget.
        It acts as a pass-through node for an integer constrained by specific steps.
        """
        # Protection against None if the note is connected incorrectly
        if num_frames is None:
            return (self.default_value,)

        return (num_frames,)

class FloatSelectorNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": ("FLOAT", {"default": 1.0,
                                    "min": 0.0,
                                    "max": 1.0,
                                    "step": 0.01})
            }
        }

    RETURN_TYPES = ("FLOAT",)        
    RETURN_NAMES = ("float",)        
    DESCRIPTION = "A simple utility node that outputs a single float value with adjustable range and precision. Useful for creating custom selectors like CFG strength."
    FUNCTION = "run"               
    CATEGORY = "utils"      

    def run(self, float):
       
        return (float,) 
        
class DA_LatentLoader:
    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        files = []
        for root, _, filenames in os.walk(output_dir):
            for f in filenames:
                if f.endswith(('.latent', '.safetensors')):
                    rel = os.path.relpath(os.path.join(root, f), output_dir).replace("\\", "/")
                    files.append(rel)
        if not files:
            files = ["[No latents found in output]"]
        return {"required": {"latent_file": (sorted(files),)}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load_latent"
    DESCRIPTION = "Loads previously saved latent tensors (.latent or .safetensors) from the current output directory. Automatically detects and parses both modern JSON-wrapped formats (supporting float32 tensors) and legacy binary formats (pickle/torch/safetensors). Ensures 4D tensor shape compatibility by broadcasting missing dimensions. Logs loaded tensor statistics (shape, min/max/mean) for debugging."
    CATEGORY = "latent"
    
    def load_latent(self, latent_file):
        if latent_file == "[No latents found in output]":
            raise FileNotFoundError("No .latent/.safetensors files found in output")
        full_path = os.path.join(folder_paths.get_output_directory(), latent_file)
        with open(full_path, "rb") as f:
            data = f.read()
        json_start = data.find(b'{')
        if json_start == -1:
            return ({"samples": self._legacy_load(data)},)
        json_end = self._find_json_end(data, json_start)
        if json_end == -1:
            raise RuntimeError("Failed to find end of JSON")
        metadata = json.loads(data[json_start:json_end].decode('utf-8'))
        if "latent_tensor" in metadata:
            info = metadata["latent_tensor"]
            shape = info["shape"]
            if info["dtype"] != "F32":
                raise RuntimeError(f"Unsupported dtype: {info['dtype']}")
            offsets = info["data_offsets"]
            # Skip spaces after JSON
            data_start = json_end
            while data_start < len(data) and data[data_start] in (0x20, 0x0A, 0x0D, 0x09):
                data_start += 1
            size = offsets[1] - offsets[0]
            raw = data[data_start : data_start + size]
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).clone()
            print(f"[LatentLoaderBrowser] {latent_file}: shape={tensor.shape}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")
            return ({"samples": tensor},)
        # fallback to old formats
        tensor = self._legacy_load(data[json_end:])
        if tensor is not None:
            return ({"samples": tensor},)
        raise RuntimeError(f"Failed to load latent from {full_path}")
    def _find_json_end(self, data, start):
        i = start
        depth = 0
        in_string = False
        while i < len(data):
            ch = data[i]
            if in_string:
                if ch == ord('\\'):
                    i += 1
                elif ch == ord('"'):
                    in_string = False
            else:
                if ch == ord('"'):
                    in_string = True
                elif ch == ord('{'):
                    depth += 1
                elif ch == ord('}'):
                    depth -= 1
                    if depth == 0:
                        return i + 1
            i += 1
        return -1
    def _legacy_load(self, blob):
        import io, pickle
        for loader in [
            lambda b: pickle.loads(b),
            lambda b: torch.load(io.BytesIO(b), map_location='cpu'),
            lambda b: __import__('safetensors.torch', fromlist=['load']).load(b) if len(b) > 0 else None,
        ]:
            try:
                obj = loader(blob)
                return self._extract(obj)
            except:
                continue
        return None
    def _extract(self, obj):
        if isinstance(obj, dict):
            t = obj.get("samples", next((v for v in obj.values() if isinstance(v, torch.Tensor)), None))
            if t is None:
                raise RuntimeError("Dictionary without tensor")
        elif isinstance(obj, torch.Tensor):
            t = obj
        else:
            raise RuntimeError(f"Unknown type: {type(obj)}")
        if t.ndim == 3:
            t = t.unsqueeze(0)
        elif t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        return t.to(torch.float32)

class DA_TiledUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "image": ("IMAGE",),
                "preview_freq": ("INT", {"default": 1, "min": 1, "max": 100, "tooltip": "How often to update the preview (e.g., 2 for every other step)"}),
                "force_full_tiles": ("BOOLEAN", {"default": True, "tooltip": "Pad edge tiles to full tile_size for consistent denoise effect"}),
            },
            "optional": {
                "upscale_model_opt": ("UPSCALE_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_process"
    DESCRIPTION = "Tiled diffusion upscaling with optional ESRGAN pre-scaling. Memory-efficient processing of large images in tiles, featuring seamless overlap blending, configurable denoise strength, and preview callbacks for detail enhancement without VRAM exhaustion."
    CATEGORY = "Image Processing/Upscale"
    
    def upscale_full_image(self, image, target_size, upscale_model_opt):
        new_H, new_W = target_size
        if upscale_model_opt is not None and UPSCALE_AVAILABLE:
            upscaler_node = ImageUpscaleWithModel()
            upscaled = upscaler_node.upscale(upscale_model_opt, image)[0]
            if upscaled.shape[1] != new_H or upscaled.shape[2] != new_W:
                upscaled = upscaled.permute(0, 3, 1, 2)
                upscaled = F.interpolate(upscaled, size=(new_H, new_W), mode='bicubic', align_corners=False)
                upscaled = upscaled.permute(0, 2, 3, 1)
        else:
            img_nchw = image.permute(0, 3, 1, 2)
            up_nchw = F.interpolate(img_nchw, size=(new_H, new_W), mode='bicubic', align_corners=False)
            upscaled = up_nchw.permute(0, 2, 3, 1)
        return upscaled

    # Process a single tile through VAE -> KSampler -> VAE decode with callback support for progress and preview.
    def diffuse_tile(self, tile, vae, model, positive, negative, seed, steps, cfg, denoise, sampler_name, scheduler, callback):
        """Process a single tile via VAE encode, diffusion sampling, and VAE decode with callback support for progress updates and previews."""
        # VAE encode (on CPU)
        latent = vae.encode(tile)
        device = comfy.model_management.get_torch_device()
        latent = latent.to(device)
        noise = comfy.sample.prepare_noise(latent, seed, None)
        # Sampling with passed callback, disabling built-in progress bar
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent, denoise=denoise,
            disable_pbar=True,  # Disable standard progress bar, use custom callback instead
            callback=callback
        )
        samples = samples.cpu()
        decoded = vae.decode(samples)
        return decoded

    def make_weight_mask(self, h, w, overlap):
        mask = torch.ones(h, w, dtype=torch.float32)
        if overlap > 0:
            top = min(overlap, h)
            if top > 0:
                mask[:top, :] *= torch.linspace(0, 1, top).view(-1, 1)
            bottom = min(overlap, h)
            if bottom > 0:
                mask[-bottom:, :] *= torch.linspace(0, 1, bottom).flip(0).view(-1, 1)
            left = min(overlap, w)
            if left > 0:
                mask[:, :left] *= torch.linspace(0, 1, left)
            right = min(overlap, w)
            if right > 0:
                mask[:, -right:] *= torch.linspace(0, 1, right).flip(0)
        return mask.unsqueeze(0).unsqueeze(-1)

    # Helper: pad tile to full size, safely choosing padding mode
    def pad_to_full(self, tile, target_h, target_w):
        _, orig_h, orig_w, _ = tile.shape
        pad_bottom = target_h - orig_h
        pad_right = target_w - orig_w
        if pad_bottom == 0 and pad_right == 0:
            return tile
        # Convert to (B, C, H, W)
        tile_nchw = tile.permute(0, 3, 1, 2)
        
        # Determine padding mode:
        # 'reflect' is nicer but requires pad < dim size for each padded dimension
        # Use 'replicate' if any padding is too large or if dimension is too small
        use_reflect = True
        if pad_bottom > 0 and (pad_bottom >= orig_h):
            use_reflect = False
        if pad_right > 0 and (pad_right >= orig_w):
            use_reflect = False
        # Also if any dimension is 1, reflect fails
        if orig_h == 1 or orig_w == 1:
            use_reflect = False
        
        mode = 'reflect' if use_reflect else 'replicate'
        # Pad order: (left, right, top, bottom) for (W, H)
        padded_nchw = F.pad(tile_nchw, (0, pad_right, 0, pad_bottom), mode=mode)
        # Convert back to (B, H, W, C)
        padded_tile = padded_nchw.permute(0, 2, 3, 1)
        return padded_tile

    # --------------------------------------------------------------------------
    # Main Process Method
    # --------------------------------------------------------------------------
    def upscale_process(self, image, vae, upscale_factor, tile_size, overlap, seed,
                        steps, cfg, denoise, positive, negative, sampler_name, scheduler,
                        model, upscale_model_opt=None, preview_freq=1, force_full_tiles=False):
        
        torch.manual_seed(seed)
        B, H, W, C = image.shape
        new_H = int(H * upscale_factor)
        new_W = int(W * upscale_factor)
        print(f"[TiledUpscaler] Target size: {new_H}x{new_W}")
        
        # ANSI color codes for terminal output
        COLOR_WARNING = "\033[93m"    # yellow
        COLOR_RESET = "\033[0m"
        COLOR_INFO = "\033[92m"       # green optional
        COLOR_ERROR = "\033[91m"      # red

        # Checking whether the tile size exceeds the final image
        if not force_full_tiles and (new_H < tile_size or new_W < tile_size):
            print(f"{COLOR_WARNING}[TiledUpscaler] WARNING: Tile size ({tile_size}) exceeds upscaled image dimensions ({new_H}x{new_W}) while 'force_full_tiles' is disabled. This will cause tensor size mismatches. Consider enabling 'force_full_tiles', reducing tile_size, or increasing upscale_factor.{COLOR_RESET}")
            force_full_tiles = True
            print(f"{COLOR_WARNING}[TiledUpscaler] Automatically enabling 'force_full_tiles' to continue processing.{COLOR_RESET}")
        
        # 1. Pre-scale the entire image first
        print("[TiledUpscaler] Pre-upscaling the full image...")
        full_upscaled = self.upscale_full_image(image, (new_H, new_W), upscale_model_opt)
        
        # 2. Tile parameters
        tile_sz = tile_size
        overlap_px = overlap
        step = max(tile_sz - overlap_px, 1)

        # 3. Calculate grid of tiles
        tiles_y = max(1, math.ceil((new_H - overlap_px) / step))
        tiles_x = max(1, math.ceil((new_W - overlap_px) / step))
        total_tiles = tiles_y * tiles_x
        print(f"[TiledUpscaler] Grid: {tiles_y}x{tiles_x} = {total_tiles} tiles, tile size={tile_sz}, overlap={overlap_px}, step={step}")

        # Global progress bar for all steps across all tiles
        total_steps_all = total_tiles * steps
        pbar = comfy.utils.ProgressBar(total_steps_all)

        # Initialize previewer for generating previews from latents
        device = comfy.model_management.get_torch_device()
        try:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)
            preview_format = "JPEG"
        except Exception as e:
            print(f"[TiledUpscaler] Preview not available: {e}")
            previewer = None

        output_canvas = torch.zeros((B, new_H, new_W, C), dtype=image.dtype, device="cpu")
        weight_canvas = torch.zeros((B, new_H, new_W, C), dtype=image.dtype, device="cpu")
        current_tile = 0

        for ty in range(tiles_y):
            y = ty * step
            y_end = min(y + tile_sz, new_H)
            for tx in range(tiles_x):
                x = tx * step
                x_end = min(x + tile_sz, new_W)
                orig_h = y_end - y
                orig_w = x_end - x
                tile = full_upscaled[:, y:y_end, x:x_end, :]
                print(f"[Tile {current_tile+1}/{total_tiles}] original size={orig_h}x{orig_w}")
                
                # Pad to full tile size if requested and needed
                if force_full_tiles and (orig_h != tile_sz or orig_w != tile_sz):
                    padded_tile = self.pad_to_full(tile, tile_sz, tile_sz)
                    process_tile = padded_tile
                else:
                    process_tile = tile
                
                # Calculate the starting global step for this tile
                start_global_step = current_tile * steps

                # Create callback for updating global progress and previews
                def make_callback(start_step, total_steps, freq, pbar_ref, previewer_ref, fmt, tile_idx, total_tiles, steps_local):
                    def callback(step, x0, x, total_steps_local):
                        global_step = start_step + step + 1
                        preview_bytes = None
                        if previewer_ref is not None and ((step + 1) % freq == 0 or (step + 1) == total_steps_local):
                            preview_bytes = previewer_ref.decode_latent_to_preview_image(fmt, x0)
                        pbar_ref.update_absolute(global_step, total_steps, preview_bytes)
                        percent = (step + 1) / steps_local
                        bar_len = 30
                        filled = int(bar_len * percent)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        sys.stdout.write(f"\rTile {tile_idx+1}/{total_tiles}: {bar} {percent*100:3.0f}% step {step+1}/{steps_local}")
                        if step + 1 == steps_local:
                            sys.stdout.write("\n")
                        sys.stdout.flush()
                    return callback

                # Instantiate callback
                callback_func = make_callback(
                    start_global_step, total_steps_all, preview_freq,
                    pbar, previewer, preview_format,
                    current_tile, total_tiles, steps
                )
                
                # Process tile (may be padded)
                refined_full = self.diffuse_tile(process_tile, vae, model, positive, negative,
                                                 seed, steps, cfg, denoise, sampler_name, scheduler,
                                                 callback=callback_func)
                
                # If tile was padded, crop back to original dimensions
                if force_full_tiles and (orig_h != tile_sz or orig_w != tile_sz):
                    refined = refined_full[:, :orig_h, :orig_w, :]
                else:
                    refined = refined_full
                
                # Smooth blending with mask based on original tile size
                mask = self.make_weight_mask(orig_h, orig_w, overlap_px)
                mask = mask.to(refined.device)
                output_canvas[:, y:y_end, x:x_end, :] += refined * mask
                weight_canvas[:, y:y_end, x:x_end, :] += mask
                current_tile += 1

        final_image = (output_canvas / (weight_canvas + 1e-8)).clamp(0, 1)
        return (final_image,)