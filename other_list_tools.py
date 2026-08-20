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
    CATEGORY = "Utils"
    DESCRIPTION = (
            "Calculates and returns the total number of elements in an input list."
            "Provides a quick way to determine batch sizes, list lengths, or count available inputs within your workflow logic."
    )

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
    DESCRIPTION = (
        "Merges two input lists into a single combined list, preserving the order of elements from both sources."
        "Essential for chaining multiple generated lists (e.g., samplers or encoders) into one cohesive pipeline stage."
    )

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
