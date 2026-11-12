import math
import os
import sys
from typing import Any

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
import torch
from torch.nn import functional as F
from ultralytics import YOLO

import nodes

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
    DESCRIPTION = (
        "An advanced KSampler supporting Model Shift (SDE), dynamic noise masking, and customizable preview frequency. "
        "Allows users to control how often latent previews are rendered during the sampling process to optimize memory usage and visual feedback."
    )

    # ----------Helper method----------
    def _apply_shift(self, model: Any, shift: float, multiplier: float = 1.0):
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
               model: Any,
               seed: int,
               steps: int,
               cfg: float,
               sampler_name: str,
               scheduler: str,
               positive: Any,
               negative: Any,
               latent_image: Any,
               denoise: float = 1.0,
               shift: float = 0.0,
               preview_freq: int = 1):
                    
        if shift:
            try:
                model = self._apply_shift(model, shift)
            except Exception as e: # noqa: BLE001
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


try:
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
    UPSCALE_AVAILABLE = True
except ImportError:
    UPSCALE_AVAILABLE = False
    print("[TiledESRGANUpscaler] Warning: comfy_extras.nodes_upscale_model not found. Upscale model will be ignored.") 

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
    CATEGORY = "Image Processing/Upscale"
    DESCRIPTION = (
        "Tiled diffusion upscaling with optional ESRGAN pre-scaling. "
        "Memory-efficient processing of large images in tiles, featuring seamless overlap blending, configurable denoise strength, and preview callbacks for detail enhancement without VRAM exhaustion."
    )
    
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
        except Exception as e: # noqa: BLE001
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


# Register or find the directory for YOLO models
def get_yolo_models():
    # Search in standard ComfyUI folders (ultralytics/bbox or just ultralytics)
    models = []
    for folder_name in ["ultralytics", "ultralytics_bbox"]:
        if folder_name in folder_paths.folder_names_and_paths:
            models.extend(folder_paths.get_filename_list(folder_name))
    return sorted(set(models)) if models else ["none"]

def get_yolo_path(model_name):
    for folder_name in ["ultralytics", "ultralytics_bbox"]:
        if folder_name in folder_paths.folder_names_and_paths:
            path = folder_paths.get_full_path(folder_name, model_name)
            if path and os.path.exists(path):
                return path
    return None

# ==============================================================================
# 1. "THIN CLIENT" (Node for building the task stack)
# ==============================================================================
class DA_RefineTaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yolo_model": (get_yolo_models(),),
                "confidence": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.05}),
                "crop_padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8, "tooltip": "Padding around the bounding box in pixels for context"}),
                "target_size": ("INT", {"default": 512, "min": 256, "max": 1536, "step": 64, "tooltip": "Length of the longest side of the crop during generation"}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_denoise": (["fixed", "smaller_less", "smaller_more"], {"default": "fixed", "tooltip": "Denoise progression based on object area"}),
                "seed_shift": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "tooltip": "Seed shift for this task (0 = use server seed)"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
            "optional": {
                "prev_stack": ("DA_REFINE_STACK",),
            }
        }

    RETURN_TYPES = ("DA_REFINE_STACK",)
    RETURN_NAMES = ("refine_stack",)
    FUNCTION = "add_task"
    CATEGORY = "Image Processing/Refiner"
    DESCRIPTION = (
        "Defines a specific refinement task by combining a YOLO model selection with parameters like denoising strength, crop size, and padding. "
        "It allows you to build a stack of multiple detection-based refinement tasks to be processed in sequence."
    )    

    def add_task(self, yolo_model, confidence, crop_padding, target_size, denoise, scale_denoise, seed_shift, positive, negative, prev_stack=None):
        # Copy the previous stack to avoid mutating data during graph re-runs
        stack = list(prev_stack) if prev_stack is not None else []
        
        if yolo_model != "none":
            task = {
                "yolo_model": yolo_model,
                "confidence": confidence,
                "crop_padding": crop_padding,
                "target_size": target_size,
                "denoise": denoise,
                "scale_denoise": scale_denoise,
                "seed_shift": seed_shift,
                "positive": positive,
                "negative": negative,
            }
            stack.append(task)
            
        return (stack,)

# ==============================================================================
# 2. "SERVER" (Execution node)
# ==============================================================================
class DA_MasterRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "mask_feather": ("INT", {"default": 16, "min": 0, "max": 128, "step": 4}),
                "preview_freq": ("INT", {"default": 2, "min": 0, "max": 100, "tooltip": "Preview update frequency in steps (0 = disable preview for max speed)"}),
            },
            "optional": {
                "refine_stack": ("DA_REFINE_STACK", {"tooltip": "Optional task stack. If disconnected or empty, acts as a pass-through (reroutes image unchanged)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_stack"
    CATEGORY = "Image Processing/Refiner"
    DESCRIPTION = (
        "Executes the batch processing of all defined tasks by detecting objects via YOLO, isolating them, and refining them using a targeted sampling process. "
        "It automatically handles cropping, resizing, and seamless blending of the refined results back into the original image. "
        "If refine_stack is disconnected or empty, passes the input image through unchanged."
    )

    # Your clean function for smooth mask edges (adapted for rectangular crops)
    def make_feather_mask(self, h, w, feather):
        mask = torch.ones((h, w), dtype=torch.float32)
        if feather > 0:
            f_h = min(feather, h // 2)
            f_w = min(feather, w // 2)
            if f_h > 0:
                mask[:f_h, :] *= torch.linspace(0, 1, f_h).view(-1, 1)
                mask[-f_h:, :] *= torch.linspace(0, 1, f_h).flip(0).view(-1, 1)
            if f_w > 0:
                mask[:, :f_w] *= torch.linspace(0, 1, f_w)
                mask[:, -f_w:] *= torch.linspace(0, 1, f_w).flip(0)
        return mask.unsqueeze(0).unsqueeze(-1) # [1, H, W, 1]

    # Calculate target resolution while preserving proportions and multiples of 8 for VAE
    def get_aspect_safe_size(self, orig_h, orig_w, target_max_side):
        if orig_h >= orig_w:
            scale = target_max_side / orig_h
            new_h = target_max_side
            new_w = max(64, round((orig_w * scale) / 8.0) * 8)
        else:
            scale = target_max_side / orig_w
            new_w = target_max_side
            new_h = max(64, round((orig_h * scale) / 8.0) * 8)
        return new_h, new_w

    def calculate_dynamic_denoise(self, base_denoise, mode, box_area, total_area):
        if mode == "fixed" or total_area == 0:
            return base_denoise
        
        # Relative area of the object (e.g., from 0.01 to 0.5 of the whole frame)
        ratio = min(max(box_area / total_area, 0.01), 0.5) / 0.5 # Normalize to [0.02 .. 1.0]
        
        if mode == "smaller_less":
            # The smaller the object, the softer the impact (from 40% to 100% of base_denoise)
            factor = 0.4 + 0.6 * ratio
        elif mode == "smaller_more":
            # The smaller the object, the stronger the redraw (from 100% to 40%)
            factor = 1.0 - 0.6 * ratio
            
        return round(min(max(base_denoise * factor, 0.05), 1.0), 3)

    # Detect objects via YOLO
    def detect_boxes_for_batch(self, images_tensor, model_name, confidence):
        model_path = get_yolo_path(model_name)
        if not model_path:
            print(f"[DA_Refiner] Warning: Model not found: {model_name}")
            return [[] for _ in range(images_tensor.shape[0])]
        
        yolo = YOLO(model_path)
        batch_boxes = []
        
        # Run the detector over each image in the batch
        for b in range(images_tensor.shape[0]):
            img_np = (images_tensor[b:b+1].cpu().numpy() * 255).clip(0, 255).astype("uint8")[0]
            results = yolo.predict(img_np, conf=confidence, verbose=False)
            boxes = []
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    boxes.append([int(coord) for coord in box])
            batch_boxes.append(boxes)
            
        del yolo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return batch_boxes

    def process_stack(self, image, model, vae, seed, steps, cfg, sampler_name, scheduler, mask_feather, preview_freq, refine_stack=None):
        if not refine_stack:
            return (image,)

        # Work with a copy of the image, updating it step by step
        canvas = image.clone()
        B, H, W, _C = canvas.shape
        total_area = H * W
        device = comfy.model_management.get_torch_device()

        # 1. Pre-pass: gather the work plan to build an accurate progress bar
        # Structure: execution_plan[task_idx] = [boxes_img_0, boxes_img_1, ...]
        execution_plan = []
        total_objects_count = 0
        
        for task in refine_stack:
            batch_boxes = self.detect_boxes_for_batch(canvas, task["yolo_model"], task["confidence"])
            execution_plan.append(batch_boxes)
            for boxes in batch_boxes:
                total_objects_count += len(boxes)

        if total_objects_count == 0:
            print("[DA_Refiner] No objects detected across all tasks. Skipping refinement.")
            return (canvas,)

        total_steps_all = total_objects_count * steps
        pbar = comfy.utils.ProgressBar(total_steps_all)
        print(f"[DA_Refiner] Found {total_objects_count} object(s) across {len(refine_stack)} task(s). Total steps: {total_steps_all}")

        # Initialize previewer only if preview_freq > 0
        previewer = None
        preview_format = "JPEG"
        if preview_freq > 0:
            try:
                previewer = latent_preview.get_previewer(device, model.model.latent_format)
            except Exception as e: # noqa: BLE001
                print(f"[DA_Refiner] Preview init failed: {e}")

        def make_callback(start_step, total_steps, freq, pbar_ref, previewer_ref, fmt, obj_idx, total_objs, steps_local):
            def callback(step, x0, x, total_steps_local):
                global_step = start_step + step + 1
                preview_bytes = None
                if previewer_ref is not None and freq > 0 and ((step + 1) % freq == 0 or (step + 1) == total_steps_local):
                    preview_bytes = previewer_ref.decode_latent_to_preview_image(fmt, x0)
                pbar_ref.update_absolute(global_step, total_steps, preview_bytes)
                
                percent = (step + 1) / steps_local
                bar_len = 25
                filled = int(bar_len * percent)
                bar = '█' * filled + '░' * (bar_len - filled)
                sys.stdout.write(f"\r[DA_Refiner] Obj {obj_idx+1}/{total_objs}: {bar} {percent*100:3.0f}% (step {step+1}/{steps_local})")
                if step + 1 == steps_local:
                    sys.stdout.write("\n")
                sys.stdout.flush()
            return callback

        current_obj_idx = 0
        # 2. Main execution loop
        for task_idx, task in enumerate(refine_stack):
            batch_boxes = execution_plan[task_idx]
            shift = task.get("seed_shift", 0)
            
            for b in range(B):
                boxes = batch_boxes[b]
                if not boxes:
                    print(f"[DA_Refiner] No objects found for {task['yolo_model']}.")
                    continue
                for box_idx, (x1, y1, x2, y2) in enumerate(boxes):
                    pad = task["crop_padding"]
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(W, x2 + pad)
                    cy2 = min(H, y2 + pad)
                    orig_w = cx2 - cx1
                    orig_h = cy2 - cy1
                    if orig_w < 16 or orig_h < 16:
                        current_obj_idx += 1
                        continue
                    # Extract crop
                    crop = canvas[b:b+1, cy1:cy2, cx1:cx2, :]
                
                    # Calculate denoise for specific object
                    box_area = (x2 - x1) * (y2 - y1)
                    current_denoise = self.calculate_dynamic_denoise(
                        task["denoise"], task["scale_denoise"], box_area, total_area
                    )

                    # Scale crop to target_size (preserving proportions or square)
                    scaled_h, scaled_w = self.get_aspect_safe_size(orig_h, orig_w, task["target_size"])
                    crop_nchw = crop.permute(0, 3, 1, 2)
                    scaled_nchw = F.interpolate(crop_nchw, size=(scaled_h, scaled_w), mode="bicubic", align_corners=False)
                    scaled_crop = scaled_nchw.permute(0, 2, 3, 1)

                    # --- Sampling Block ---
                    # Sampling with callback
                    start_global_step = current_obj_idx * steps
                    callback_func = make_callback(
                        start_global_step, total_steps_all, preview_freq,
                        pbar, previewer, preview_format,
                        current_obj_idx, total_objects_count, steps
                    )
                    latent = vae.encode(scaled_crop).to(device)
                    # Account for client seed_shift with a large step to avoid collisions between objects
                    current_seed = (seed + (shift * 100000) + (b * 10000) + (task_idx * 1000) + box_idx) & 0xffffffffffffffff
                    noise = comfy.sample.prepare_noise(latent, current_seed, None)
                    
                    samples = comfy.sample.sample(
                        model, noise, steps, cfg, sampler_name, scheduler,
                        task["positive"], task["negative"], latent, denoise=current_denoise,
                        disable_pbar=True,
                        callback=callback_func
                    )
                    
                    decoded = vae.decode(samples.cpu())
                    # -------------------------

                    # Return crop to original size [orig_h, orig_w]
                    dec_nchw = decoded.permute(0, 3, 1, 2)
                    restored_nchw = F.interpolate(dec_nchw, size=(orig_h, orig_w), mode="bicubic", align_corners=False)
                    restored_crop = restored_nchw.permute(0, 2, 3, 1)

                    # Stitch back with soft edges
                    mask = self.make_feather_mask(orig_h, orig_w, mask_feather).to(canvas.device)
                    original_patch = canvas[b:b+1, cy1:cy2, cx1:cx2, :]
                    blended_patch = original_patch * (1.0 - mask) + restored_crop * mask
                    canvas[b:b+1, cy1:cy2, cx1:cx2, :] = blended_patch
                    current_obj_idx += 1
        return (canvas,)
