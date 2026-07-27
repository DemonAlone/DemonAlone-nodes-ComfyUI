import os
import json
import hashlib
import fnmatch
import logging
from datetime import datetime
from pathlib import Path
import comfy.utils
import torch
import numpy as np
from torch.nn import functional as F
from PIL import Image, ImageOps
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile
from PIL.ExifTags import TAGS
import piexif

import folder_paths

logger = logging.getLogger(__name__)

# ImageResizeNode is based on  Image Resize from https://github.com/cubiq/ComfyUI_essentials

class ImageResizeNode:
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


#This node is based on CImageLoadWithMetadata from https://github.com/crystian/ComfyUI-Crystools
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


class SaveImageNoMetaNode:
    """
    Saves an image without workflow/metadata.
    Supports %date% placeholder which is replaced by yyyy-mm-dd or a custom mask.
    Supports png and jpg formats.
    Always saves the file inside the `output/<relative path>` folder.
    An index is added automatically (00001, 00002 ...).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "path": ("STRING", {"default": "ComfyUI", "tooltip": "Relative path inside `output`. Use %date% for date."}),
                "format": (["png", "jpg"],),
                "preview": ("BOOLEAN", {"default": True}),
                "date_mask": ("BOOLEAN", {"default": False, "label_on": "Custom Date", "label_off": "Default Date"}),
                "custom_date_format": ("STRING", {"default": "yyyy-mm-dd", "tooltip": "Date format if custom mask is enabled (e.g., yyyy, dd-mm-yyyy)"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "ImageSaver"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves images to the output folder without embedding workflow metadata. Supports PNG/JPEG formats, automatic date stamping (%date%) with custom format support, and auto-indexing (e.g., 00001.png) for duplicate files."

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

    def save(self, image, path: str, format: str, preview: bool, date_mask: bool, custom_date_format: str):
        if not path:
            raise ValueError("Save path is not specified")
            
        # Determine date string based on toggle
        current_date = datetime.now()
        date_str = current_date.strftime("%Y-%m-%d")
        
        if date_mask:
            # Parse custom format (simple mapping)
            year = str(current_date.year).zfill(4)
            month = str(current_date.month).zfill(2)
            day = str(current_date.day).zfill(2)
            
            # Simple replacement for common tokens
            date_str = custom_date_format.replace("yyyy", year).replace("mm", month).replace("dd", day)

        # Replace %date% with the calculated string
        processed_path = path.replace("%date%", date_str)
        
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

