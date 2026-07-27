class DA_AudioDebugNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "debug_audio"
    CATEGORY = "debug"
    DESCRIPTION = "Outputs audio parameters: number of channels, sample rate, duration, tensor shape."

    def debug_audio(self, audio):
        if audio is None:
            return ("Audio is None",)
        
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        
        if waveform is None:
            return ("Audio has no waveform",)
        
        # Get dimensions
        shape = tuple(waveform.shape)
        # Number of channels – usually second parameter (batch, channels, samples) or (channels, samples)
        if len(shape) == 3:
            batch, channels, samples = shape
        elif len(shape) == 2:
            channels, samples = shape
            batch = 1
        else:
            return (f"Unexpected shape: {shape}",)
        
        duration = samples / sample_rate if sample_rate else 0.0
        
        info = (
            f"Shape: {shape}\n"
            f"Channels: {channels}\n"
            f"Sample rate: {sample_rate} Hz\n"
            f"Samples: {samples}\n"
            f"Duration: {duration:.2f} s"
        )
        return (info,)

class MaskDebugNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "debug"
    DESCRIPTION = "Inspects and reports the tensor shape of a connected mask node in string format. Useful for debugging pipeline issues related to dimension mismatches or verifying input consistency before further processing steps."

    def debug(self, mask):
        import torch
        t = mask.squeeze(-1) if mask.ndim == 4 and mask.shape[3] == 1 else mask
        return (f"shape={tuple(t.shape)}",)