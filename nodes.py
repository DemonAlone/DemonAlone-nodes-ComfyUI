import os
import json
import re
import itertools
import torch
import folder_paths
from nodes import VAEDecode, PreviewImage


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
        
class PatchModelSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "patch_name": (folder_paths.get_filename_list("model_patches"),),
            }
        }

    RETURN_TYPES = (folder_paths.get_filename_list("model_patches"), "STRING")
    RETURN_NAMES = ("patch_name", "patch_name_str")
    CATEGORY = "utils"
    DESCRIPTION = "Retrieves patch model names from the ComfyUI patches folder and outputs both filename tuple (combo) and string representation for dynamic patch selection."
    FUNCTION = "get_patch_name"

    def get_patch_name(self, patch_name):
        return patch_name, patch_name

class MultiPlaceholderPromptList:
    """
    Generates a list of prompts, replacing multiple placeholders in a template with all possible combinations of values from corresponding lists.
    Placeholders in the template should be specified in the form {name}.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Template": ("STRING", {
                    "multiline": True,
                    "default": "a girl in a {color} dress with {hair} hair",  
                    "tooltip": "Main prompt template. Use placeholders like {name}. Additional placeholders can be added below; all are equivalent and optional."
                }),
                "values_separator": ("STRING", {
                    "default": ",",
                    "tooltip": "Separator used for listing values (supports \\n). Use \\n for multi-line lists."
                }),
                "placeholder1": ("STRING", {
                    "default": "{color}",
                    "tooltip": "First placeholder to replace. Optional and equivalent to other placeholders."
                }),
                "values1": ("STRING", {
                    "multiline": True,
                    "default": "blue, red, black",
                    "tooltip": "Comma-separated list of values for the first placeholder. Use '_empty_' as a value to insert an empty string."
                }),
                "placeholder2": ("STRING", {
                    "default": "{hair}",
                    "tooltip": "Second placeholder to replace. Optional and equivalent to others."
                }),
                "values2": ("STRING", {
                    "multiline": True,
                    "default": "blonde, brown, black",
                    "tooltip": "Comma-separated list of values for the second placeholder. Use '_empty_' as a value to insert an empty string."
                }),
                "placeholder3": ("STRING", {
                    "default": "",
                    "tooltip": "Third placeholder (optional)."
                }),
                "values3": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Values for the third placeholder. Use '_empty_' as a value to insert an empty string in the final prompt."
                }),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "generate"
    CATEGORY = "utility/text"
    DESCRIPTION = "Creates a list of prompts, substituting all possible combinations of values from specified lists into placeholders in the template."

    def generate(self, Template, values_separator, placeholder1, values1, placeholder2, values2, placeholder3, values3):
        # Separator processing (supports \n)
        sep = "\n" if values_separator == r"\n" else values_separator

        # Collect pairs (placeholder, value list), ignoring empty ones
        pairs = []
        for ph, vals in [(placeholder1, values1), (placeholder2, values2), (placeholder3, values3)]:
            if ph.strip() and vals.strip():
                # Splitting the string into elements, trimming edge whitespace, and removing empty items caused by accidental line breaks

                raw_list = [v.strip() for v in vals.split(sep) if v.strip()]
                
                value_list = []
                for v in raw_list:
                    if v == "_empty_":
                        value_list.append("")  # Convert marker to a valid empty string
                    else:
                        value_list.append(v)
                            
                    if value_list:   
                            pairs.append((ph.strip(), value_list))

        # If no pairs at all — return cleaned original template
        if not pairs:
            return ([Template.strip()],)

        # Find all placeholders in template (searching for {something})
        found = re.findall(r'\{[^}]+\}', Template)
        unique_in_template = []
        for ph in found:
            if ph not in unique_in_template:
                unique_in_template.append(ph)

        # Dictionary: placeholder → value list
        ph_to_values = dict(pairs)

        # Take only those placeholders that are in the template and have values defined
        vary_pairs = [(ph, ph_to_values[ph]) for ph in unique_in_template if ph in ph_to_values]

        # If none exist — return cleaned template
        if not vary_pairs:
            return ([Template.strip()],)

        # Generate Cartesian product
        value_lists = [values for _, values in vary_pairs]
        combinations = list(itertools.product(*value_lists))

        # Collect results
        results = []
        for combo in combinations:
            prompt = Template
            for (ph, _), val in zip(vary_pairs, combo):
                prompt = prompt.replace(ph, val)
            
        # Remove unnecessary double spaces left where the placeholder was replaced, and trim the string ends from accidental newlines
            cleaned_prompt = " ".join(prompt.split())
            results.append(cleaned_prompt.strip())

        return (results,)

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
    DESCRIPTION = "Decodes latents to images using the provided VAE when enabled, outputting both Latent and Image for previewing. When disabled, acts as a pass-through (Reroute) node, passing the original Latent unchanged without decoding."
    OUTPUT_NODE = True

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
