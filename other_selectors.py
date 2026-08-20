import folder_paths

class BooleanSwitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {        
                "state": ("BOOLEAN",)
            },
            "optional": {          
                "on_true": ("*",), 
                "on_false": ("*",)
            }
        }

    RETURN_TYPES = ("*",)    
    FUNCTION = "switch"      
    CATEGORY = "logic"    
    DESCRIPTION = "A conditional routing node that outputs either the 'true' or 'false' input value based on the current boolean state."

    def switch(self, state: bool, on_true=None, on_false=None):
        if state:
            return (on_true if on_true is not None else None,)
        else:
            return (on_false if on_false is not None else None,)

class FloatSelectorNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": ("FLOAT", {"default": 1.0,
                                    "min": 0.0,
                                    "max": 1.0,
                                    "step": 0.01})
            }
        }

    RETURN_TYPES = ("FLOAT",)    
    RETURN_NAMES = ("float",)
    FUNCTION = "run"
    CATEGORY = "utils"
    DESCRIPTION = "A simple utility node that outputs a single float value with adjustable range and precision. Useful for creating custom selectors like CFG strength."

    def run(self, float):
       
        return (float,)

class PonyPrefixesNode:
    """
    Score     – 5 variants
        "-"               → None
        "Everything"      → "score_9, score_8_up, score_7_up, score_6_up, score_5_up, "
        "Average"         → "score_9, score_8_up, score_7_up, score_6_up, score_5_up, "
        "Good"            → "score_9, score_8_up, score_7_up, "
        "Only the best"   → "score_9, "

    Rating    – 4 variants
        "-"               → None
        "Safe"            → "rating_safe, "
        "Questionable"    → "rating_questionable, "
        "Explicit"        → "rating_explicit, "

    Source    – 5 variants
        "-"               → None
        "Anime"           → "source_anime, "
        "Furry"           → "source_furry, "
        "Cartoon"         → "source_cartoon, "
        "Pony"            → "source_pony, "

   In results combined string (order: Score → Rating → Source).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 1 list – Score
                "score": (
                    [
                        "-",
                        "Everything",
                        "Average",
                        "Good",
                        "Only the best",
                    ],
                ),
                # 2 list – Rating
                "rating": (
                    [
                        "-",
                        "Safe",
                        "Questionable",
                        "Explicit",
                    ],
                ),
                # 3 list – Source
                "source": (
                    [
                        "-",
                        "Anime",
                        "Furry",
                        "Cartoon",
                        "Pony",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_string",)   # output name
    FUNCTION = "generate"
    CATEGORY = "utils"
    DESCRIPTION = "Dynamically constructs a prompt prefix string containing Pony-specific score, rating, and source tags (e.g., score_9, rating_safe). Combines selected variants into a single comma-separated string for immediate use in text encoding."
    
    # mappings
    _SCORE_MAP = {
        "Everything":     "score_9, score_8_up, score_7_up, score_6_up, score_5_up, ",
        "Average":        "score_9, score_8_up, score_7_up, score_6_up, score_5_up, ",
        "Good":           "score_9, score_8_up, score_7_up, ",
        "Only the best":  "score_9, ",
    }

    _RATING_MAP = {
        "Safe":          "rating_safe, ",
        "Questionable":  "rating_questionable, ",
        "Explicit":      "rating_explicit, ",
    }

    _SOURCE_MAP = {
        "Anime":   "source_anime, ",
        "Furry":   "source_furry, ",
        "Cartoon": "source_cartoon, ",
        "Pony":    "source_pony, ",
    }

    def generate(self, score="-", rating="-", source="-"):
        """Comnining whole string"""
        parts = []

        if score != "-":
            parts.append(self._SCORE_MAP.get(score, ""))

        if rating != "-":
            parts.append(self._RATING_MAP.get(rating, ""))

        if source != "-":
            parts.append(self._SOURCE_MAP.get(source, ""))

        result = "".join(parts)

        return (result,)

class ShiftSliderNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shift": ("FLOAT", {"default": 0.0,
                                    "min": 0.0,
                                    "max": 100.0,
                                    "step": 0.01})
            }
        }

    RETURN_TYPES = ("FLOAT",)        
    RETURN_NAMES = ("shift",)        
    DESCRIPTION = "Applies a dynamic shift value to the sampling process or subsequent operations."
    FUNCTION = "run"               
    CATEGORY = "utils"      

    def run(self, shift):
       
        return (shift,)    
        
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
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1})
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("clip_skip",)
    FUNCTION = "get_value"           # method that will be invoked
    CATEGORY = "utils"
    DESCRIPTION = "Outputs an integer clip skip value ranging from -24 to -1. Provides fine-grained control over the depth of CLIP token skipping in diffusion models."

    def get_value(self, clip_skip):
        """
        Receives the slider's current integer value and returns it.
        The return is wrapped in a tuple because the node interface expects
        an iterable of outputs.
        """
        return (clip_skip,)

class WanNumFramesNode:
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("num_frames",)
    FUNCTION = "execute"
    CATEGORY = "utils"
    DESCRIPTION = "Output integer value with strict range constraints (min: 1, max: 10000, step: 4)." 


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_frames": ("INT", {"default": 49, 
                                   "min": 1, 
                                   "max": 10000, 
                                   "step": 4}), 
            }
        }

    def execute(self, num_frames=None):
        """
        Returns the integer value provided by the input widget.
        It acts as a pass-through node for an integer constrained by specific steps.
        """
        # Protection against None if the note is connected incorrectly
        if num_frames is None:
            return (self.default_value,)

        return (num_frames,)

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
    FUNCTION = "get_patch_name"
    DESCRIPTION = "Retrieves patch model names from the ComfyUI patches folder and outputs both filename tuple (combo) and string representation for dynamic patch selection."

    def get_patch_name(self, patch_name):
        return patch_name, patch_name