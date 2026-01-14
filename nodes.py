import comfy.samplers
import folder_paths 


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
        inputs = {"required": {f"diff_model_{i+1}": (models, {"default": "none"}) for i in range(10)}} # 10 слотов
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
