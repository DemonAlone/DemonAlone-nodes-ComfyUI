import torch
import numpy
import numpy as np
from comfy_execution.graph_utils import GraphBuilder
from comfy_execution.graph import ExecutionBlocker
import math
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS, GPSTAGS, IFD         
from PIL.PngImagePlugin import PngImageFile
import os
import folder_paths
import time

def images_grid_by_x(
    images: list,
    max_x_items: int,
    gap: int = 0,
    bg_color: tuple = (0, 0, 0),
    ) -> Image.Image:
    """
    Creates a grid layout from a list of PIL images based on X axis (columns).
    max_x_items - maximum number of items in a row (axis X)
    gap - spacing between items
    bg_color - background color for the grid
    """
    if not images:
        raise ValueError("List of images is empty")

    n = len(images)
    x_count = min(max_x_items, n) # Limit to maximum allowed X items
    y_count = (n + x_count - 1) // x_count # Limit to maximum allowed Y rows

    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)

    total_w = x_count * cell_w + (x_count - 1) * gap
    total_h = y_count * cell_h + (y_count - 1) * gap
    
    grid = Image.new("RGB", (total_w, total_h), bg_color)
    
    # Place each image into its appropriate position in the grid
    for idx, img in enumerate(images):
        y_idx = idx // x_count
        x_idx = idx % x_count
        x = x_idx * (cell_w + gap)
        y = y_idx * (cell_h + gap)
        # Center the image within its cell
        offset_x = (cell_w - img.width) // 2
        offset_y = (cell_h - img.height) // 2
        grid.paste(img, (x + offset_x, y + offset_y))

    return grid

#feedbackNode, MyXYZHelper, MyXYGridAccumulator, MyXYZSuperStacker nodes based on nodes from https://github.com/kenjiqq/qq-nodes-comfyui
# --- BASE CLASS FOR PREVIEW ---
class FeedbackNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    def preview_images(self, images, filename_prefix="MyXYZ"):
        if not images or len(images) == 0:
            return []
        
        # Get save parameters from the first image
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        
        results = list()
        for image in images:
            # Convert tensor to array for PIL
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        return results

# --- 1. HELPER ---
class MyXYZHelper:
    _last_index = -1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x_list": ("LIST",),
                "y_list": ("LIST",),
                "z_list": ("LIST",),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "x_prefix": ("STRING", {"default": ""}),
                "y_prefix": ("STRING", {"default": ""}),
                "z_prefix": ("STRING", {"default": ""}),
                "font_size": ("INT", {"default": 30, "min": 10, "max": 40}),
                "grid_gap": ("INT", {"default": 20, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "XYZ_GRID_CONTROL")
    RETURN_NAMES = ("x_value", "y_value", "z_value", "XYZ_GRID_CONTROL")
    FUNCTION = "run"
    CATEGORY = "Utils"
    DESCRIPTION = "Orchestrates the grid layout by mapping the current execution index to specific X, Y, and Z values. Dynamically generates annotations for the XYZ plot headers based on input lists and styling parameters."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def run(self, x_list, y_list, z_list, index, **kwargs):
        force_reset = (index == 0) or (index < self._last_index)
        self._last_index = index

        len_x, len_y, len_z = len(x_list), len(y_list), len(z_list)
        total_per_page = len_x * len_y
        
        z_idx = (index // total_per_page) % len_z
        adj_idx = index % total_per_page
        
        y_idx = adj_idx % len_y
        x_idx = (adj_idx // len_y) % len_x

        x_pre = kwargs.get('x_prefix', "")
        y_pre = kwargs.get('y_prefix', "")
        z_pre = kwargs.get('z_prefix', "")

        x_ann = ";".join([f"{x_pre}: {str(v)}" if x_pre else str(v) for v in x_list])
        y_ann = ";".join([f"{y_pre}: {str(v)}" if y_pre else str(v) for v in y_list])
        z_label = f"{z_pre}: {str(z_list[z_idx])}" if z_pre else str(z_list[z_idx])

        XYZ_GRID_CONTROL = (
            total_per_page, 
            0 if force_reset else adj_idx, 
            y_ann, # Passing vertical headers
            x_ann, # Passing horizontal headers
            len_x, # Grid width (number of columns along the X axis)
            kwargs.get('font_size', 50), 
            kwargs.get('grid_gap', 20),
            z_label,
            z_idx,
            len_z,
            0 if force_reset else index
        )

        return (x_list[x_idx], y_list[y_idx], z_list[z_idx], XYZ_GRID_CONTROL)

# --- 2. ACCUMULATOR---
class MyXYGridAccumulator(FeedbackNode):
    image_batch = torch.Tensor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "XYZ_GRID_CONTROL": ("XYZ_GRID_CONTROL",),
                "show_previews": ("BOOLEAN", {"default": True}),
                "max_preview_mp": ("FLOAT", {"default": 15.0, "min": 4.0, "max": 18.0, "step": 0.1, "tooltip": "Megapixel limit for resized preview. If total pixels exceed this, the page is scaled down"}),
                "max_preview_side": ("INT", {"default": 8192, "min": 2048, "max": 16384, "step": 128, "tooltip": "Maximum width/height in pixels for resized preview. If either side exceeds this, the page is scaled down."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("full_page", "resized_page")
    FUNCTION = "run"
    CATEGORY = "Utils"
    DESCRIPTION = "Buffers individual images into a visual grid as the XYZ loop progresses. Handles the accumulation of images per page, renders the preview grid with axis labels when full, and clears the batch upon completion of each slice."

    def run(self, images, XYZ_GRID_CONTROL, show_previews, max_preview_mp, max_preview_side, unique_id):
        # Unpacking controls
        count, reset_val, y_txt, x_txt, x_size, f_size, gap, z_label, *_ = XYZ_GRID_CONTROL
        
        # --- 1. Batch accumulation ---
        if reset_val == 0:
            MyXYGridAccumulator.image_batch = images
        else:
            if MyXYGridAccumulator.image_batch.numel() == 0:
                MyXYGridAccumulator.image_batch = images
            else:
                MyXYGridAccumulator.image_batch = torch.cat((
                    MyXYGridAccumulator.image_batch, 
                    images.to(MyXYGridAccumulator.image_batch.device)
                ), dim=0)
        
        curr_num = MyXYGridAccumulator.image_batch.shape[0]

        # If the page is not yet filled, block execution (and optionally show previews of individual cells)
        if curr_num < count:
            ui_res = []
            if show_previews:
                preview_list = [MyXYGridAccumulator.image_batch[i] for i in range(curr_num)]
                ui_res = self.preview_images(preview_list)
            return {"result": (ExecutionBlocker(None), ExecutionBlocker(None)), "ui": {"images": ui_res}}

        # --- 2. the page is full - create a grid ---
        page_imgs = MyXYGridAccumulator.image_batch[:count]
        MyXYGridAccumulator.image_batch = torch.Tensor()  # Clear for the next page

        # Convert tensors to PIL.Image
        pil_images = []
        for i in range(count):
            img_tensor = page_imgs[i]
            np_img = (255. * img_tensor.cpu().numpy()).astype(np.uint8)
            pil_images.append(Image.fromarray(np_img))

        # Prepare font
        try:
            font = ImageFont.truetype("arial.ttf", f_size)
            font_big = ImageFont.truetype("arial.ttf", int(f_size * 1.5))
        except:
            font = ImageFont.load_default()
            font_big = font

        # --- 3. Basic grid (cells only, no labels) ---
        base_grid = images_grid_by_x(pil_images, max_x_items=x_size, gap=gap)
        base_w, base_h = base_grid.size
        
        # --- 4. Function for drawing labels on the passed grid ---
        def add_labels_to_grid(grid_img, x_texts, y_texts, z_label, font, font_big, gap, cell_w, cell_h, y_count, x_count, max_y_width=400):
            img = grid_img.copy()
            draw = ImageDraw.Draw(img)
            
            # Settings
            x_text_padding = 5
            y_text_padding = 10
            x_line_spacing = 5
            y_line_spacing = 3               # intra-line spacing for Y labels

            # --- Text wrapping function by words ---
            def wrap_text(text, font, max_width, draw):
                words = text.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = f"{current_line} {word}".strip()
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    w = bbox[2] - bbox[0]
                    if w <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        # Too long word — split character by character
                        if draw.textbbox((0, 0), word, font=font)[2] > max_width:
                            sub_line = ""
                            for ch in word:
                                if draw.textbbox((0, 0), sub_line + ch, font=font)[2] <= max_width:
                                    sub_line += ch
                                else:
                                    lines.append(sub_line)
                                    sub_line = ch
                            if sub_line:
                                current_line = sub_line
                            else:
                                current_line = ""
                        else:
                            current_line = word
                if current_line:
                    lines.append(current_line)
                return lines if lines else [""]

            # X labels
            x_wrapped = []
            x_label_height = 0
            if x_texts:
                for xt in x_texts:
                    lines = wrap_text(xt, font, cell_w - 2 * x_text_padding, draw)
                    x_wrapped.append(lines)
                    line_h = draw.textbbox((0, 0), "Ay", font=font)[3] - draw.textbbox((0, 0), "Ay", font=font)[1]
                    needed_h = len(lines) * (line_h + x_line_spacing) + x_text_padding
                    if needed_h > x_label_height:
                        x_label_height = needed_h

            # Y labels
            y_wrapped = []
            if y_texts:
                line_h = draw.textbbox((0, 0), "Ay", font=font)[3] - draw.textbbox((0, 0), "Ay", font=font)[1]
                max_lines = max(1, (cell_h - 2 * y_text_padding) // (line_h + y_line_spacing))
                for yt in y_texts:
                    lines = wrap_text(yt, font, max_y_width - 2 * y_text_padding, draw)
                    if len(lines) > max_lines:
                        lines = lines[:max_lines]
                        last_line = lines[-1] + '…'
                        while draw.textbbox((0, 0), last_line, font=font)[2] > max_y_width - 2 * y_text_padding and len(last_line) > 1:
                            last_line = last_line[:-4] + '…'
                        lines[-1] = last_line if last_line else '…'
                    y_wrapped.append(lines)

            # Z label
            z_height = 0
            if z_label:
                z_height = int(f_size * 1.5) + 20

            # Final canvas  with labels ---
            final_w = max_y_width + grid_img.width
            final_h = z_height + x_label_height + grid_img.height
            final_img = Image.new("RGB", (final_w, final_h), color=(0, 0, 0))
            final_draw = ImageDraw.Draw(final_img)

            # Draw Z label ---
            if z_label:
                bbox = final_draw.textbbox((0, 0), z_label, font=font_big)
                zw = bbox[2] - bbox[0]
                zh = bbox[3] - bbox[1]
                final_draw.text(((final_w - zw) // 2, (z_height - zh) // 2), z_label, font=font_big, fill=(255, 255, 255))

            # Draw X axis labels ---
            if x_texts:
                x_start_x = max_y_width
                for idx, lines in enumerate(x_wrapped):
                    x_center = x_start_x + idx * (cell_w + gap) + cell_w // 2
                    y = z_height + x_text_padding
                    line_h = final_draw.textbbox((0, 0), "Ay", font=font)[3] - final_draw.textbbox((0, 0), "Ay", font=font)[1]
                    for line in lines:
                        lw = final_draw.textbbox((0, 0), line, font=font)[2]
                        final_draw.text((x_center - lw // 2, y), line, font=font, fill=(255, 255, 255))
                        y += line_h + x_line_spacing

            # Draw Y axis labels ---
            if y_texts:
                y_start = z_height + x_label_height + gap // 2
                for idx, lines in enumerate(y_wrapped):
                    y_center = y_start + idx * (cell_h + gap) + cell_h // 2
                    line_h = final_draw.textbbox((0, 0), "Ay", font=font)[3] - final_draw.textbbox((0, 0), "Ay", font=font)[1]
                    total_text_h = len(lines) * (line_h + y_line_spacing) - y_line_spacing
                    current_y = y_center - total_text_h // 2
                    for line in lines:
                        lw = final_draw.textbbox((0, 0), line, font=font)[2]
                        x = (max_y_width - lw) // 2
                        final_draw.text((x, current_y), line, font=font, fill=(255, 255, 255))
                        current_y += line_h + y_line_spacing

            # Paste image grid ---
            final_img.paste(grid_img, (max_y_width, z_height + x_label_height))
            # Replace grid_img ---
            return final_img

        # --- 5. Creating a full-size page with labels ---
        # Parsing lists of signatures
        x_texts_list = x_txt.split(";") if x_txt else []
        y_texts_list = y_txt.split(";") if y_txt else []
        y_count = len(y_texts_list) if y_texts_list else 1
        x_count = len(x_texts_list) if x_texts_list else 1
        cell_w_full = pil_images[0].width
        cell_h_full = pil_images[0].height

        full_page_img = add_labels_to_grid(
            base_grid, x_texts_list, y_texts_list, z_label,
            font, font_big, gap,
            cell_w_full, cell_h_full, y_count, x_count,
            max_y_width=400
        )

        # --- 6. Resize the entire page if needed (for preview) ---
        full_w, full_h = full_page_img.size
        full_pixels = full_w * full_h
        max_pixels = max_preview_mp * 1_000_000

        # Calculate the scale taking into account pixel and side restrictions
        scale = 1.0
        if full_pixels > max_pixels:
            scale = min(scale, (max_pixels / full_pixels) ** 0.5)
        if full_w > max_preview_side:
            scale = min(scale, max_preview_side / full_w)
        if full_h > max_preview_side:
            scale = min(scale, max_preview_side / full_h)

        if scale < 1.0:
            new_w = int(full_w * scale)
            new_h = int(full_h * scale)
            resized_page_img = full_page_img.resize((new_w, new_h), Image.LANCZOS)
        else:
            resized_page_img = full_page_img

        # --- 7. Conversion to tensors ---
        def pil_to_tensor(pil_img):
            np_arr = np.array(pil_img).astype(np.float32) / 255.0
            return torch.from_numpy(np_arr).unsqueeze(0)  # [1, H, W, C]

        full_tensor = pil_to_tensor(full_page_img)
        preview_tensor = pil_to_tensor(resized_page_img)

        # --- 8. UI preview (individual cells) ---
        ui_res = []
        if show_previews:
            ui_res = self.preview_images([page_imgs[i] for i in range(count)])

        return {"result": (full_tensor, preview_tensor), "ui": {"images": ui_res}}   
        
# --- 3. SUPER STACKER (final batch) ---
class MyXYZSuperStacker:
    storage = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "full_page": ("IMAGE",),
                "resized_page": ("IMAGE",),
                "XYZ_GRID_CONTROL": ("XYZ_GRID_CONTROL",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("full_stack", "resized_stack")
    FUNCTION = "stack"
    CATEGORY = "Utils"
    DESCRIPTION = "Collects all rendered XY pages into a single multi-page image sequence once the full XYZ dataset is generated. Acts as the final aggregation node to output the complete result set for saving or further processing."

    def stack(self, full_page, resized_page, XYZ_GRID_CONTROL):
        *_, z_idx, total_z, g_index = XYZ_GRID_CONTROL
        # Reset on first call (g_index == 0)
        if g_index == 0:
            self.storage_full = []
            self.storage_preview = []

        # In case the attributes were not created for some reason (for example, g_index is not 0 the first time it is called)
        if not hasattr(self, 'storage_full'):
            self.storage_full = []
            self.storage_preview = []

        if len(self.storage_full) == z_idx:
            self.storage_full.append(full_page)
            self.storage_preview.append(resized_page)

        if len(self.storage_full) >= total_z:
            full_stack = torch.cat(self.storage_full, dim=0)
            resized_stack = torch.cat(self.storage_preview, dim=0)
            self.storage_full = []
            self.storage_preview = []
            return (full_stack, resized_stack)
        else:
            return (ExecutionBlocker(None), ExecutionBlocker(None))
     
class XYZConflictValidatorAndSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_type": (["STRING", "INT", "FLOAT", "*"],),
                "global_val": ("*",), 
            },
            "optional": {
                "x": ("*",),
                "y": ("*",),
                "z": ("*",),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "utils/XYZ"
    DESCRIPTION = "Safeguards the XYZ pipeline against type mismatches by enforcing that only one active parameter exists per execution step. Automatically casts and outputs the current grid value (X, Y, or Z) as an Int, Float, or String based on configuration."
    
    def execute(self, output_type, global_val, **kwargs):
        # 1. Multiple choice check (Axis conflict)
        active_inputs = {k: v for k, v in kwargs.items() if v is not None}
        
        if len(active_inputs) > 1:
            raise ValueError(f"⚠️ XYZ Conflict: Few active parameters of the same type: {list(active_inputs.keys())}")

        # Choice value (XYZ или Global)
        raw_result = active_inputs[next(iter(active_inputs))] if active_inputs else global_val

        # 2. Type casting according to the selection in the menu
        try:
            if output_type == "INT":
                return (int(float(raw_result)),)
            elif output_type == "FLOAT":
                return (float(raw_result),)
            elif output_type == "STRING":
                return (str(raw_result),)
            else:
                return (raw_result,)
        except (ValueError, TypeError):
            raise TypeError(f"⚠️ XYZ Type Mismatch: Unable to turn '{raw_result}' into a {output_type}")