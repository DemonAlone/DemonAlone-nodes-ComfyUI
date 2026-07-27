import torch
from torch.nn import functional as F
import comfy.samplers
import comfy.model_management
import comfy.sample
import latent_preview
import nodes
import torchvision.transforms as transforms
import comfy.model_management
import math
import sys


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