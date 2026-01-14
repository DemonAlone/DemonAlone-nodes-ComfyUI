from .nodes import (
    SamplerGeneratorNode,
    SchedulerGeneratorNode,
    ModelGeneratorNode,
    DiffusionModelGeneratorNode
)

NODE_CLASS_MAPPINGS = {
    "SamplerGeneratorNode": SamplerGeneratorNode,
    "SchedulerGeneratorNode": SchedulerGeneratorNode,
    "ModelGeneratorNode": ModelGeneratorNode,
    "DiffusionModelGeneratorNode": DiffusionModelGeneratorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerGeneratorNode": "Dynamic Sampler Selector",
    "SchedulerGeneratorNode": "Dynamic Scheduler Selector",
    "ModelGeneratorNode": "Dynamic Checkpoint Selector",
    "DiffusionModelGeneratorNode": "Dynamic Diffusion Model Selector",
}
