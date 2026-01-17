import comfy.samplers
import folder_paths 
import os


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
       # modified node from Mikey-nodes by bash-j
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
