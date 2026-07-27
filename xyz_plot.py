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

def images_grid_by_columns(
    images: list,
    max_columns: int,
    gap: int = 0,
    bg_color: tuple = (0, 0, 0),
    ) -> Image.Image:
    """
    Creates a grid layout from a list of PIL images.
    max_columns - maximum number of columns in the grid
    gap - spacing between columns
    bg_color - background color for the grid
    """
    if not images:
        raise ValueError("List of images is empty")

    n = len(images)
    columns = min(max_columns, n) # Limit to maximum allowed columns
    rows = (n + columns - 1) // columns # Limit to maximum allowed columns

    # Limit to maximum allowed columns
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)

    total_w = columns * cell_w + (columns - 1) * gap
    total_h = rows * cell_h + (rows - 1) * gap
    # Limit to maximum allowed columns
    grid = Image.new("RGB", (total_w, total_h), bg_color)
    
    # Place each image into its appropriate position in the grid
    for idx, img in enumerate(images):
        row = idx // columns
        col = idx % columns
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)
        # Center the image within its cell
        offset_x = (cell_w - img.width) // 2
        offset_y = (cell_h - img.height) // 2
        grid.paste(img, (x + offset_x, y + offset_y))

    return grid

#feedbackNode, MyXYZHelper, MyXYGridAccumulator,  MyXYZSuperStacker nodes based on nodes from  https://github.com/kenjiqq/qq-nodes-comfyui
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
                "row_list": ("LIST",),
                "column_list": ("LIST",),
                "page_list": ("LIST",),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "row_prefix": ("STRING", {"default": ""}),
                "column_prefix": ("STRING", {"default": ""}),
                "page_prefix": ("STRING", {"default": ""}),
                "font_size": ("INT", {"default": 20, "min": 10, "max": 40}),
                "grid_gap": ("INT", {"default": 20, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "XYZ_GRID_CONTROL")
    RETURN_NAMES = ("row_value", "column_value", "page_value", "XYZ_GRID_CONTROL")
    FUNCTION = "run"
    CATEGORY = "Utils"
    DESCRIPTION = "Orchestrates the grid layout by mapping the current execution index to specific row, column, and page values. Dynamically generates annotations (e.g., 'Value: 5') for the XYZ plot headers based on input lists and styling parameters."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def run(self, row_list, column_list, page_list, index, **kwargs):
        force_reset = (index == 0) or (index < self._last_index)
        self._last_index = index

        len_x, len_y, len_z = len(column_list), len(row_list), len(page_list)
        total_per_page = len_x * len_y
        
        z_idx = (index // total_per_page) % len_z
        adj_idx = index % total_per_page
        row_idx = (adj_idx // len_x) % len_y
        col_idx = adj_idx % len_x

        r_pre = kwargs.get('row_prefix', "")
        c_pre = kwargs.get('column_prefix', "")
        p_pre = kwargs.get('page_prefix', "")

        row_ann = ";".join([f"{r_pre}: {str(x)}" if r_pre else str(x) for x in row_list])
        col_ann = ";".join([f"{c_pre}: {str(y)}" if c_pre else str(y) for y in column_list])
        page_label = f"{p_pre}: {str(page_list[z_idx])}" if p_pre else str(page_list[z_idx])

        XYZ_GRID_CONTROL = (
            total_per_page, 
            0 if force_reset else adj_idx, 
            row_ann, 
            col_ann, 
            len_x, 
            kwargs.get('font_size', 50), 
            kwargs.get('grid_gap', 20),
            page_label,
            z_idx,
            len_z,
            0 if force_reset else index
        )

        return (row_list[row_idx], column_list[col_idx], page_list[z_idx], XYZ_GRID_CONTROL)

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
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Utils"
    DESCRIPTION = "Buffers individual images into a visual grid as the XYZ loop progresses. Handles the accumulation of images per page, renders the preview grid with axis labels when full, and clears the batch upon completion of each slice."

    def run(self, images, XYZ_GRID_CONTROL, show_previews, unique_id):
        count, reset_val, row_txt, col_txt, x_size, f_size, gap, z_label, *_ = XYZ_GRID_CONTROL
        
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

        if curr_num < count:
            ui_res = []
            if show_previews:
                preview_list = [MyXYGridAccumulator.image_batch[i] for i in range(curr_num)]
                ui_res = self.preview_images(preview_list)
            return {"result": (ExecutionBlocker(None),), "ui": {"images": ui_res}}

        page_imgs = MyXYGridAccumulator.image_batch[:count]
        MyXYGridAccumulator.image_batch = torch.Tensor()  # Clear for the next page

        # 1. Convert tensors to PIL.Image
        pil_images = []
        for i in range(count):
            img_tensor = page_imgs[i]
            np_img = (255. * img_tensor.cpu().numpy()).astype(np.uint8)
            pil_images.append(Image.fromarray(np_img))

        # 2. Prepare font
        try:
            font = ImageFont.truetype("arial.ttf", f_size)
        except:
            font = ImageFont.load_default()

        # 3. Build the main image grid (rows × columns)
        grid_img = images_grid_by_columns(pil_images, max_columns=x_size, gap=gap)

        draw_temp = ImageDraw.Draw(grid_img)

        # Settings
        max_row_width = 400              # fixed width of left border (adjustable)
        row_text_padding = 10
        col_text_padding = 5
        col_line_spacing = 5
        row_line_spacing = 3             # intra-row line spacing

        row_texts_list = row_txt.split(";") if row_txt else []
        col_texts_list = col_txt.split(";") if col_txt else []

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

        # --- 1. Prepare column labels (wrap by width) ---
        col_wrapped = []
        col_label_height = 0
        if col_texts_list:
            col_width = pil_images[0].width + gap if pil_images else grid_img.width // x_size
            for ct in col_texts_list:
                lines = wrap_text(ct, font, col_width - 2 * col_text_padding, draw_temp)
                col_wrapped.append(lines)
                line_h = draw_temp.textbbox((0, 0), "Ay", font=font)[3] - draw_temp.textbbox((0, 0), "Ay", font=font)[1]
                needed_h = len(lines) * (line_h + col_line_spacing) + col_text_padding
                if needed_h > col_label_height:
                    col_label_height = needed_h

        # --- 2. Prepare row labels (wrap with height limit) ---
        row_wrapped = []          # list of lists of lines for each row label
        if row_texts_list:
            # Height of one image row (including gap)
            row_h_total = pil_images[0].height + gap
            # Height of one text line
            line_h = draw_temp.textbbox((0, 0), "Ay", font=font)[3] - draw_temp.textbbox((0, 0), "Ay", font=font)[1]
            # Maximum number of lines that fit in row_h_total
            max_lines = max(1, (row_h_total - 2 * row_text_padding) // (line_h + row_line_spacing))

            for rt in row_texts_list:
                lines = wrap_text(rt, font, max_row_width - 2 * row_text_padding, draw_temp)
                if len(lines) > max_lines:
                    # Keep only max_lines rows, truncate the last one with ellipsis
                    lines = lines[:max_lines]
                    # Truncate the last line to fit width and add '…'
                    last_line = lines[-1] + '…'
                    while draw_temp.textbbox((0, 0), last_line, font=font)[2] > max_row_width - 2 * row_text_padding and len(last_line) > 1:
                        last_line = last_line[:-4] + '…'
                    lines[-1] = last_line if last_line else '…'
                row_wrapped.append(lines)
        else:
            row_wrapped = []

        # --- 3. z_label (page header) ---
        z_height = 0
        if z_label:
            try:
                font_big = ImageFont.truetype("arial.ttf", int(f_size * 1.5))
            except:
                font_big = ImageFont.load_default()
            z_height = int(f_size * 1.5) + 20

        # --- 4. Final canvas ---
        final_w = max_row_width + grid_img.width
        final_h = z_height + col_label_height + grid_img.height
        final_img = Image.new("RGB", (final_w, final_h), color=(0, 0, 0))
        draw_final = ImageDraw.Draw(final_img)

        # --- 5. z_label ---
        if z_label:
            bbox = draw_final.textbbox((0, 0), z_label, font=font_big)
            zw = bbox[2] - bbox[0]
            zh = bbox[3] - bbox[1]
            draw_final.text(((final_w - zw) // 2, (z_height - zh) // 2), z_label, font=font_big, fill=(255, 255, 255))

        # --- 6. column labels ---
        if col_texts_list:
            col_start_x = max_row_width
            img_w = pil_images[0].width
            for idx, lines in enumerate(col_wrapped):
                x_center = col_start_x + idx * (img_w + gap) + img_w // 2
                y = z_height + col_text_padding
                line_h = draw_final.textbbox((0, 0), "Ay", font=font)[3] - draw_final.textbbox((0, 0), "Ay", font=font)[1]
                for line in lines:
                    lw = draw_final.textbbox((0, 0), line, font=font)[2]
                    draw_final.text((x_center - lw // 2, y), line, font=font, fill=(255, 255, 255))
                    y += line_h + col_line_spacing

        # --- 7. row labels (multi-line) ---
        if row_texts_list:
            row_h_total = pil_images[0].height + gap
            y_start = z_height + col_label_height + gap // 2
            for idx, lines in enumerate(row_wrapped):
                # Vertical center of the image row
                y_center = y_start + idx * row_h_total + row_h_total // 2
                line_h = draw_final.textbbox((0, 0), "Ay", font=font)[3] - draw_final.textbbox((0, 0), "Ay", font=font)[1]
                total_text_h = len(lines) * (line_h + row_line_spacing) - row_line_spacing
                # Start drawing so the text block is vertically centered
                current_y = y_center - total_text_h // 2
                for line in lines:
                    lw = draw_final.textbbox((0, 0), line, font=font)[2]
                    # Center text horizontally within the left border
                    x = (max_row_width - lw) // 2
                    draw_final.text((x, current_y), line, font=font, fill=(255, 255, 255))
                    current_y += line_h + row_line_spacing

        # --- 8. Paste image grid ---
        final_img.paste(grid_img, (max_row_width, z_height + col_label_height))

        # --- 9. Replace grid_img ---
        grid_img = final_img

        # 6. Convert back to ComfyUI tensor format
        final_np = np.array(grid_img).astype(np.float32) / 255.0
        final_tensor = torch.from_numpy(final_np).unsqueeze(0)  # [1, H, W, C]

        # 7. Preview (if enabled)
        ui_res = []
        if show_previews:
            # Display individual images of the current page (not the final grid)
            ui_res = self.preview_images([page_imgs[i] for i in range(count)])

        return {"result": (final_tensor,), "ui": {"images": ui_res}}

# --- 3. SUPER STACKER (final batch) ---
class MyXYZSuperStacker:
    storage = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "XYZ_GRID_CONTROL": ("XYZ_GRID_CONTROL",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "stack"
    CATEGORY = "Utils"
    DESCRIPTION = "Collects all rendered XY pages into a single multi-page image sequence once the full XYZ dataset is generated. Acts as the final aggregation node to output the complete result set for saving or further processing."

    def stack(self, image, XYZ_GRID_CONTROL):
        *_, z_idx, total_z, g_index = XYZ_GRID_CONTROL
        
        if g_index == 0:
            MyXYZSuperStacker.storage = []

        if len(MyXYZSuperStacker.storage) == z_idx:
            MyXYZSuperStacker.storage.append(image)

        if len(MyXYZSuperStacker.storage) >= total_z:
            all_pages = torch.cat(MyXYZSuperStacker.storage, dim=0)
            MyXYZSuperStacker.storage = [] 
            return (all_pages,)
        else:
            return (ExecutionBlocker(None),)
     
class XYZConflictValidatorAndSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_type": (["STRING", "INT", "FLOAT", "*"],),
                "global_val": ("*",), 
            },
            "optional": {
                "row": ("*",),
                "column": ("*",),
                "pages": ("*",),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "utils/XYZ"
    DESCRIPTION = "Safeguards the XYZ pipeline against type mismatches by enforcing that only one active parameter exists per execution step. Automatically casts and outputs the current grid value (Row, Column, or Page) as an Int, Float, or String based on configuration."
    
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
