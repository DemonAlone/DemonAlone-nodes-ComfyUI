class AnyAdapterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"input_any": ("*",)}}

    RETURN_TYPES = ("*",)
    FUNCTION = "adapt"
    CATEGORY = "utils"
    DESCRIPTION = (
        "A flexible pass-through node that accepts any input type." 
        "It safely forwards data downstream or returns a clean None output if the input is unconnected, ensuring pipeline stability without breaking custom workflows."
    )    
        
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
        
class AnytoIntegerAdapterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"input_any": ("*",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "adapt"
    CATEGORY = "utils"
    DESCRIPTION = (
        "Safely converts any compatible input value into an integer."
        "If the conversion fails or is impossible, it returns None without crashing the workflow."
    )
    
    def adapt(self, input_any):
        """
        Converts the input data to an integer. If conversion is impossible, returns None
        """
        try:
            return (int(input_any),)
        except (ValueError, TypeError):
            print(f"Cannot convert '{input_any}' to an integer.")
            return (None,)

class AnytoFloatAdapterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"input_any": ("*",)}}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "adapt"
    CATEGORY = "utils"
    DESCRIPTION = (
        "Safely converts any compatible input value into a floating-point number." 
        "If the conversion fails or is impossible, it returns None to prevent workflow errors."
    )

    def adapt(self, input_any):
        """
        Converts the input data to an integer. If conversion is impossible, returns None
        """
        try:
            return (float(input_any),)
        except (ValueError, TypeError):
            print(f"Cannot convert '{input_any}' to an integer.")
            return (None,)

class AnyConcatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            # Delimiter: always a text field, can be changed manually
            "required": {"delimiter": ("STRING", {"default": " "})},

            # text1…text5 are only connectors.
            "optional": {f"text{i}": ("STRING", {"forceInput": True}) for i in range(1, 6)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat"
    CATEGORY = "utils"
    DESCRIPTION = (
        "Concatenates any number of up to 5 text inputs into a single string using a custom delimiter." 
        "Acts as a flexible joiner that automatically ignores unconnected slots, ideal for building dynamic text prompts or combining parameters from various sources."
    )

    def concat(self, delimiter: str, **kwargs):
        """
        kwargs contains only those slots that were actually connected.
        If a slot was not connected, it simply is absent from the dict.
        """
        texts = [str(v) for v in kwargs.values() if v]
        return (delimiter.join(texts),)

class StringToAnyNode:
    """
    Universal Bridge: Takes a string and returns it as type '*'
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any_output",)
    FUNCTION = "convert"
    CATEGORY = "utils"
    DESCRIPTION = "Converts a string value into a universal type-compatible output, allowing seamless bridging between different data types."

    def convert(self, input_string):
        return (input_string,)

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
    DESCRIPTION = (
        "Converts a text input string into a floating-point number with error handling."
        "If the string cannot be parsed as a valid float, it logs the issue and returns 0.0 to keep the pipeline running smoothly."
    )
    
    def convert(self, text_value):
        try:
            return (float(text_value),)
        except Exception as e:
            print(f"[StringToFloatNode] Conversion error for '{text_value}': {e}")
            return (0.0,)

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
    DESCRIPTION = (
        "Safely converts a text input string into an integer."
        "Automatically handles parsing errors by logging them and returning 0 as a fallback, ensuring the workflow continues without crashing on invalid numeric strings."
    )
    
    def convert(self, text_value):
        try:
            return (int(text_value),)
        except Exception as e:
            # Log the error in the console; here we simply return 0
            print(f"[StringToIntNode] Conversion error for '{text_value}': {e}")
            return (0,)

# modified node TextConcat from https://github.com/bash-j/mikey_nodes
class TextConcatNode:
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
    DESCRIPTION = "Merges up to 5 optional text inputs into a single continuous string, using a configurable delimiter."

    def concat(self,
               delimiter,
               text1="", text2="", text3="",
               text4="", text5=""):
        """Collect non‑empty strings into a single text."""
        texts = [t for t in (text1, text2, text3, text4, text5) if t]
        return (delimiter.join(texts),)   