import os
import json
import re
import itertools
import torch
import folder_paths
from nodes import VAEDecode, PreviewImage


class OptionalCondMergeNode:
    """
    Smart "merge" for conditioning (collector of parallel ControlNet branches).
    - inputs: cond1, cond2, cond3 (optional)
    - output: one merged list of conditions
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
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
    DESCRIPTION = "Merges from 1 to 3 conditions (e.g., parallel ControlNet branches) into a single stream without mathematical distortion of tensors."

    def merge(self, **kwargs):
        # Collect only inputs actually connected
        conds = [c for c in (kwargs.get('cond1'),
                            kwargs.get('cond2'),
                            kwargs.get('cond3')) if c is not None]

        # Collect only inputs actually connected
        if not conds:
            return (None,)

        # If only 1 input is connected - node acts as a Reroute (passes through unchanged)
        if len(conds) == 1:
            return (conds[0],)

        # If multiple inputs - concatenate them into a single list
        # This preserves all metadata and Apply ControlNet settings intact
        merged = []
        for c in conds:
            merged.extend(c)

        return (merged,)

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
            except Exception:
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

class ConditionalVAEDecodePreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "preview": ("BOOLEAN", {"default": False, "tooltip": "Preview and Image Output"}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("LATENT", "IMAGE")
    FUNCTION = "process"
    CATEGORY = "custom_nodes"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Decodes latents to images using the provided VAE when enabled, outputting both Latent and Image for previewing."
        "When disabled, acts as a pass-through (Reroute) node, passing the original Latent unchanged without decoding."
    )

    def process(self, samples, vae, preview):
        if not preview:
            # Return None for the image and an empty UI
            return {"ui": {"images": []}, "result": (samples, None)}

        # Perform decoding
        decoder = VAEDecode()
        decoded_image = decoder.decode(vae, samples)[0]
        
        # Use the logic of the standard PreviewImage node for generating a preview
        # The save_images method in PreviewImage automatically handles display in the browser
        prev = PreviewImage()
        result = prev.save_images(images=decoded_image)
        
        # Add our generated image to the result for passing further
        return {"ui": result["ui"], "result": (samples, decoded_image)}
