class CountListNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_list": ("LIST",)
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "get_length"
    DESCRIPTION = "Calculates and returns the total number of elements in an input list. Provides a quick way to determine batch sizes, list lengths, or count available inputs within your workflow logic."
    CATEGORY = "Utils"

    def get_length(self, input_list):
        length = len(input_list)
        return (length,)

class ListCombinerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_a": ("LIST",),
                "list_b": ("LIST",) 
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "join_lists"
    CATEGORY = "List Operations"
    DESCRIPTION = "Merges two input lists into a single combined list, preserving the order of elements from both sources. Essential for chaining multiple generated lists (e.g., samplers or encoders) into one cohesive pipeline stage."
    
    def join_lists(self, list_a, list_b):
        return (list_a + list_b,)

class ListCreaterNode:
    """
    Splits the input multiline text by the specified delimiter 
    and returns a list of strings.
    Supports escaped sequences, for example "\n" → newline.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Separator": ("STRING", {"default": ",", "tooltip": "\\n → newline"}),
                "RemoveEmptyValues": ("BOOLEAN", {"default": False, "label_on": True, "label_off": False}),
                "Text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "split_text"
    CATEGORY = "utility/text"
    DESCRIPTION = "Splits text strings into a list of items using custom delimiters (supports commas, newlines, etc.) with optional empty value removal."

    def split_text(self, Text: str, Separator: str, RemoveEmptyValues: bool):
        # Logic for the "\n" separator
        if Separator == r"\n":
            sep = "\n"
        else:
            sep = Separator
        
        parts = Text.split(sep)
        cleaned = [p.strip().replace('\n', ' ') for p in parts]

        if RemoveEmptyValues:
            cleaned = [p for p in cleaned if p]

        return (cleaned,)

class ListRerouteNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_list": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "reroute"
    DESCRIPTION = "Passes a list input through unchanged. Acts as a connector or placeholder in the graph to manage node connections without altering data content."
    CATEGORY = "utils"

    def reroute(self, input_list):
        return (input_list,)

