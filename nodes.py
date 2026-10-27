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
        "Decodes latents to images using the provided VAE when enabled, outputting both Latent and Image for previewing. "
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
