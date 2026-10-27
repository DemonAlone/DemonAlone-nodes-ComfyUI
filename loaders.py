import os
import folder_paths
import comfy.sd
import os
import json
import torch
import logging
logger = logging.getLogger(__name__)

class DA_LoadDiffusionModelAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 1. Combo list that will scan the folder diffusion_models
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                # 2.Full support for choosing the accuracy of the scale (as in the original)
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16"], {"default": "default"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "model_name_str", "model_name_clean")
    FUNCTION = "load_model"
    CATEGORY = "advanced/loaders"
    DESCRIPTION = "Loads a diffusion model with advanced options to specify the precision data type (e.g., FP8, FP16, BF16) and outputs clean text identifiers alongside the model object"

    def load_model(self, model_name, weight_dtype):
        # Get the full physical path to the model file
        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Configuring Scale Data Types (dtype) Based on User Selection
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = comfy.model_management.XFORMERS_FP8_E4M3FN if hasattr(comfy.model_management, "XFORMERS_FP8_E4M3FN") else None
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = comfy.model_management.XFORMERS_FP8_E5M2 if hasattr(comfy.model_management, "XFORMERS_FP8_E5M2") else None
        elif weight_dtype == "fp16":
            model_options["dtype"] = torch.float16 if 'torch' in globals() else None
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16 if 'torch' in globals() else None

        # Clear the dictionary from None values so that ComfyUI uses its defaults
        model_options = {k: v for k, v in model_options.items() if v is not None}

        print(f"Loading custom diffusion model: {model_name} with dtype: {weight_dtype}")
        
        patched_model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)
        
        # Forming Clean Names for Text Outputs
        clean_name = os.path.splitext(os.path.basename(model_name))[0]
        
        return (patched_model, model_name, clean_name)

class DA_LoadCheckpointAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "ckpt_name_str", "ckpt_name_clean")
    FUNCTION = "load_checkpoint"
    CATEGORY = "advanced/loaders"
    DESCRIPTION = "Loads a full checkpoint configuration including the model, CLIP, and VAE while providing additional clean text names for the loaded assets"

    def load_checkpoint(self, ckpt_name):
        # Get the full physical path to the checkpoint file
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
            
        print(f"Loading custom checkpoint: {ckpt_name}")
        
        # Call the native ComfyUI function to load the checkpoint.
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        
        # Forming Clean Names for Text Outputs
        clean_name = os.path.splitext(os.path.basename(ckpt_name))[0]
        
        # Return 3 objects for a graph and 2 rows
        return (out[0], out[1], out[2], ckpt_name, clean_name)

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
    CATEGORY = "latent"
    DESCRIPTION = "Loads previously saved latent tensors from output directory safely."

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
                if ch == 92:
                    i += 1
                elif ch == 34:
                    in_string = False
            else:
                if ch == 34:
                    in_string = True
                elif ch == 123:
                    depth += 1
                elif ch == 125:
                    depth -= 1
                    if depth == 0:
                        return i + 1
            i += 1
        return -1
    def _legacy_load(self, blob):
        import io
        from safetensors.torch import load as safetensors_load

        loaders = [
            lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=True),
            lambda b: safetensors_load(b) if len(b) > 0 else None,
        ]
        
        for loader in loaders:
            try:
                obj = loader(blob)
                return self._extract(obj)
            except Exception as e:
                logger.debug(f"Legacy loader attempt failed: {e}")
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