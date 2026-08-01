import comfy.samplers
import folder_paths 
import os

def get_diffusion_model_file_list():
    diffusion_models = folder_paths.get_filename_list("diffusion_models")
    return ["none"] + diffusion_models

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


def get_sampler_list():
    return ["none"] + comfy.samplers.KSampler.SAMPLERS
    
class SamplerGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        samplers = get_sampler_list()
        inputs = {"required": {f"sampler_{i+1}": (samplers, {"default": "none"}) for i in range(14)}}
        return inputs
    RETURN_TYPES = ("STRING", "LIST")  
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    DESCRIPTION = "Generates a comma-separated string and list of any selected samplers from up to 10 inputs. Useful for dynamically constructing sampler lists based on user selection before feeding them into sampling nodes."
    def generate_string(self, **kwargs):
        selected = []
        for i in range(14):
            name = kwargs.get(f"sampler_{i+1}")
            if name and name != "none": selected.append(name)
        string_output = ", ".join(selected)
        list_output = selected  
        return (string_output, list_output)

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


def get_scheduler_list():
    return ["none"] + comfy.samplers.KSampler.SCHEDULERS

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


def get_checkpoint_list():
    checkpoints = folder_paths.get_filename_list("checkpoints")
    return ["none"] + checkpoints

class ModelGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_checkpoint_list()
        inputs = {"required": {f"model_{i+1}": (models, {"default": "none"}) for i in range(10)}}
        return inputs
    RETURN_TYPES = ("STRING","LIST")
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    DESCRIPTION = "Generates a combined list and comma-separated string of selected checkpoint files for flexible pipeline configuration."
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"model_{i+1}")
            if name and name != "none": selected.append(name)
        string_output = ", ".join(selected)
        list_output = selected  
        return (string_output, list_output)
 
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


def get_text_encoder_list():
    return ["none"] + folder_paths.get_filename_list("text_encoders")

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


def get_vae_list():
    """Returns a list of VAE files plus 'none'."""
    vae_files = folder_paths.get_filename_list("vae")   # key "vae"
    return ["none"] + vae_files

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
                    "step": 0.05,
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
