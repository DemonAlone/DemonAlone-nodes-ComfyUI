import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import folder_paths
from server import PromptServer
import logging

logger = logging.getLogger(__name__)

class ImageCounterNode:
    global_image_count = 0
    last_prompt_id = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "count_images"
    CATEGORY = "utility/logging"

    def count_images(self, images):
        current_prompt_id = None

        # Reading the server queue from memory for auto-reset

        try:
            if PromptServer.instance and hasattr(PromptServer.instance, 'prompt_queue'):
                queue_running, queue_pending = PromptServer.instance.prompt_queue.get_current_queue()
                if queue_running is not None and isinstance(queue_running, list) and len(queue_running) > 0:
                    current_prompt_id = queue_running
        except Exception as e:
            logger.debug(f"Failed to read server queue for auto-reset: {e}")

        # Reset counter on new Queue run
        if current_prompt_id is not None and current_prompt_id != ImageCounterNode.last_prompt_id:
            ImageCounterNode.global_image_count = 0
            ImageCounterNode.last_prompt_id = current_prompt_id
            print(f"\n[XYZ Plot] New run detected. Counter reset to 0.\n")

        # Explicitly take only the first element of the tuple shape (number of image in batch)
        # This ensures that batch_size is a clean INT, not a torch.Size object
        if len(images.shape) == 4:
            batch_size = int(images.shape[0])
        else:
            batch_size = 1

        ImageCounterNode.global_image_count += batch_size
        
        # Retrieves the exact multiplied number from our axis calculator stored in class memory
        # (Uses safe getattr method in case the calculator hasn't run yet)

        actual_total = getattr(NativeXYZTotalCalculator, 'final_total_grid_images', 1)

        # Prints a nice log to the console
        print(f"======================================")
        if actual_total > 1:
            print(f"[XYZ Plot] Progress: Image #{ImageCounterNode.global_image_count} of {actual_total}")
            percent = (ImageCounterNode.global_image_count / actual_total) * 100
            print(f"[XYZ Plot] Completed: {percent:.1f}%")
        else:
            print(f"[XYZ Plot] Image #: {ImageCounterNode.global_image_count}")
        print(f"======================================\n")
        
        return (images,)

class NativeXYZGridRenderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Final batch from the outermost EndLoop
                "x_labels": ("STRING",),  
                "y_labels": ("STRING",),  
                "z_labels": ("STRING",),  
            },
            "optional": {
                "x_prefix": ("STRING", {"default": "X"}),
                "y_prefix": ("STRING", {"default": "Y"}),
                "z_prefix": ("STRING", {"default": "Z"}),
                "font_size": ("INT", {"default": 30, "min": 10, "max": 60}),
                "grid_gap": ("INT", {"default": 20, "min": 0, "max": 100}),
                "max_preview_mp": ("FLOAT", {"default": 20.0, "min": 5.0, "max": 21.0, "step": 0.1}),
                "max_side_limit": ("INT", {"default": 16384, "min": 2048, "max": 16384}),
            }
        }

    INPUT_IS_LIST = True  
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid_image",)
    FUNCTION = "render_grid"
    CATEGORY = "utils/XYZ"
    DESCRIPTION = "Accepts a ready-made batch of images from the native loop and builds an XYZ grid, where each Z-plane is a separate batch page."

    def render_grid(self, images, x_labels, y_labels, z_labels, **kwargs):
        # 1. Unpacking incoming images into a single Python list
        pil_images = []
        if isinstance(images, list):
            for img_item in images:
                if len(img_item.shape) == 4:
                    for i in range(img_item.shape[0]):
                        np_img = (255. * img_item[i].cpu().numpy()).astype(np.uint8)
                        pil_images.append(Image.fromarray(np_img))
                else:
                    np_img = (255. * images.cpu().numpy()).astype(np.uint8) # Note: Fixed indentation/logic here based on standard practice if list item is 3D, but keeping original logic structure mostly intact except for the print translation request.
                    # Actually, looking at original code: `np_img = (255. * images[i].cpu().numpy())` was inside else block for single tensor case. Let's stick to exact logic but translate comments.
                    # Correction based on original logic provided in prompt:
                    if len(img_item.shape) != 4:
                         np_img = (255. * img_item.cpu().numpy()).astype(np.uint8)
                         pil_images.append(Image.fromarray(np_img))
        else:
            if len(images.shape) == 4:
                for i in range(images.shape[0]):
                    np_img = (255. * images[i].cpu().numpy()).astype(np.uint8)
                    pil_images.append(Image.fromarray(np_img))

        total_images = len(pil_images)
        if total_images == 0:
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        # Cell parameters
        cell_w = pil_images[0].width
        cell_h = pil_images[0].height

        # Unpacking settings from list inputs
        gap = kwargs.get('grid_gap', [20])[0]
        f_size = kwargs.get('font_size', [30])[0]
        max_preview_mp = kwargs.get('max_preview_mp', [20.0])[0]
        max_side_limit = kwargs.get('max_side_limit', [16384])[0]
        x_pre = kwargs.get('x_prefix', ["X"])[0]
        y_pre = kwargs.get('y_prefix', ["Y"])[0]
        z_pre = kwargs.get('z_prefix', ["Z"])[0]

        len_x = len(x_labels) if x_labels else 1
        len_y = len(y_labels) if y_labels else 1
        len_z = len(z_labels) if z_labels else 1
        
        images_per_page = len_x * len_y

        try:
            font = ImageFont.truetype("arial.ttf", f_size)
            font_big = ImageFont.truetype("arial.ttf", int(f_size * 1.5))
        except:
            font = ImageFont.load_default()
            font_big = font

        def wrap_text(text, font, max_width, draw_ctx):
            words = str(text).split()
            lines = []
            current_line = ""
            for word in words:
                test_line = f"{current_line} {word}".strip()
                bbox = draw_ctx.textbbox((0, 0), test_line, font=font)
                w = bbox[2] - bbox[0]
                if w <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            return lines if lines else [""]

        # 2. Rendering EACH Z-page as an independent PIL canvas
        pages_pil = []

        for z_idx in range(len_z):
            start_img_idx = z_idx * images_per_page
            end_img_idx = min(start_img_idx + images_per_page, total_images)
            page_pils = pil_images[start_img_idx:end_img_idx]
            
            if not page_pils:
                continue

            # Assembling XY grid for the current Z page
            x_count = min(len_x, len(page_pils))
            y_count = (len(page_pils) + x_count - 1) // x_count
            base_w = x_count * cell_w + (x_count - 1) * gap
            base_h = y_count * cell_h + (y_count - 1) * gap
            base_grid = Image.new("RGB", (base_w, base_h), (0, 0, 0))
            
            for idx, img in enumerate(page_pils):
                curr_x_idx = idx % len_x
                curr_y_idx = idx // len_x
                x_pos = curr_x_idx * (cell_w + gap) + (cell_w - img.width) // 2
                y_pos = curr_y_idx * (cell_h + gap) + (cell_h - img.height) // 2
                base_grid.paste(img, (x_pos, y_pos))

            # Preparing texts
            x_texts = [f"{x_pre}: {str(v)}" if x_pre else str(v) for v in x_labels] if x_labels else []
            y_texts = [f"{y_pre}: {str(v)}" if y_pre else str(v) for v in y_labels] if y_labels else []
            z_label = f"{z_pre}: {str(z_labels[z_idx])}" if z_pre and z_labels and z_idx < len(z_labels) else (str(z_labels[z_idx]) if z_labels and z_idx < len(z_labels) else "")

            max_y_width = 400
            temp_img = Image.new("RGB", (10, 10))
            temp_draw = ImageDraw.Draw(temp_img)

            # Calculating X header height
            x_wrapped = []
            x_label_height = 0
            for xt in x_texts:
                lines = wrap_text(xt, font, cell_w - 10, temp_draw)
                x_wrapped.append(lines)
                line_bbox = temp_draw.textbbox((0, 0), "Ay", font=font)
                line_h = line_bbox[3] - line_bbox[1]
                needed_h = len(lines) * (line_h + 5) + 10
                if needed_h > x_label_height:
                    x_label_height = needed_h

            z_height = int(f_size * 1.5) + 20 if z_label else 0

            # Creating canvas for ONE page
            final_w = max_y_width + base_grid.width
            final_h = z_height + x_label_height + base_grid.height
            page_canvas = Image.new("RGB", (final_w, final_h), color=(0, 0, 0))
            draw = ImageDraw.Draw(page_canvas)

            # Drawing Z, X, Y labels (with textbbox tuple fix)
            if z_label:
                bbox = draw.textbbox((0, 0), z_label, font=font_big)
                zw, zh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((final_w - zw) // 2, (z_height - zh) // 2), z_label, font=font_big, fill=(255, 255, 255))

            x_start_x = max_y_width
            for idx, lines in enumerate(x_wrapped):
                if idx >= len_x: break
                x_center = x_start_x + idx * (cell_w + gap) + cell_w // 2
                y_pos = z_height + 5
                line_bbox = draw.textbbox((0, 0), "Ay", font=font)
                line_h = line_bbox[3] - line_bbox[1]
                for line in lines:
                    lw_bbox = draw.textbbox((0, 0), line, font=font)
                    lw = lw_bbox[2] - lw_bbox[0]
                    draw.text((x_center - lw // 2, y_pos), line, font=font, fill=(255, 255, 255))
                    y_pos += line_h + 5

            y_start_y = z_height + x_label_height
            line_bbox = draw.textbbox((0, 0), "Ay", font=font)
            line_h = line_bbox[3] - line_bbox[1]
            for idx, yt in enumerate(y_texts):
                if idx >= len_y: break
                lines = wrap_text(yt, font, max_y_width - 20, draw)
                y_center = y_start_y + idx * (cell_h + gap) + cell_h // 2
                total_text_h = len(lines) * (line_h + 3) - 3
                current_y = y_center - total_text_h // 2
                for line in lines:
                    lw_bbox = draw.textbbox((0, 0), line, font=font)
                    lw = lw_bbox[2] - lw_bbox[0]
                    draw.text(((max_y_width - lw) // 2, current_y), line, font=font, fill=(255, 255, 255))
                    current_y += line_h + 3

            page_canvas.paste(base_grid, (max_y_width, z_height + x_label_height))
            
            # --- Smart resize for each page independently ---
            p_w, p_h = page_canvas.size
            p_pixels = p_w * p_h
            max_pixels = max_preview_mp * 1000000

            scale = 1.0
            if p_pixels > max_pixels:
                scale = min(scale, (max_pixels / p_pixels) ** 0.5)
            if p_w > max_side_limit:
                scale = min(scale, max_side_limit / p_w)
            if p_h > max_side_limit:
                scale = min(scale, max_side_limit / p_h)

            if scale < 1.0:
                new_w = int(p_w * scale)
                new_h = int(p_h * scale)
                print(f"[XYZ Page {z_idx}] Page resized under limit: {new_w}x{new_h} (Scale: {scale:.2f})")
                page_canvas = page_canvas.resize((new_w, new_h), Image.LANCZOS)

            pages_pil.append(page_canvas)

        # 3. Converting pages into a single comfyui tensor batch
        tensor_list = []
        for p in pages_pil:
            np_arr = np.array(p).astype(np.float32) / 255.0
            tensor_list.append(torch.from_numpy(np_arr).unsqueeze(0)) # [1, H, W, C]

        # Gluing pages along the first dimension (batch)
        output_batch = torch.cat(tensor_list, dim=0)

        return (output_batch,)

class NativeXYZTotalCalculator:
    # Static global variable where ONE final number of frames will be stored
    final_total_grid_images = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x_axis_list": ("STRING",), 
                "y_axis_list": ("STRING",),
                "z_axis_list": ("STRING",),
            }
        }

    INPUT_IS_LIST = True # Force the kernel to give us all three lists entirely at once
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("total_images",)
    FUNCTION = "calculate_total"
    CATEGORY = "utility/text/native"
    DESCRIPTION = "Accepts three native lists of axes, multiplies their lengths and saves the total volume of the Plot for the logger."

    def calculate_total(self, x_axis_list, y_axis_list, z_axis_list):
        # Clean and safe calculation of X-axis length
        if isinstance(x_axis_list, list):
            len_x = len(x_axis_list)
        else:
            len_x = 1

        # Clean and safe calculation of Y axis length (WITHOUT walrus operators)
        if isinstance(y_axis_list, list):
            len_y = len(y_axis_list)
        else:
            len_y = 1

        # Clean and safe calculation of Z axis length 
        if isinstance(z_axis_list, list):
            len_z = len(z_axis_list)
        else:
            len_z = 1

        # Multiply the active axes to obtain the exact amount of generation
        total = len_x * len_y * len_z
        
        if total < 1:
            total = 1

        # save ONE resulting number into class memory
        NativeXYZTotalCalculator.final_total_grid_images = total
        
        print(f"\n======================================")
        print(f"[XYZ Calculator] Grid calculation successful!")
        print(f"[XYZ Calculator] Formula: X({len_x}) * Y({len_y}) * Z({len_z}) = {total} images.")
        print(f"======================================\n")
        
        return (total,)
