import torch
import numpy as np
import cv2
from PIL import Image
import math

# Helper functions (copied or adapted from insert_anything_node.py)
def standardize_and_binarize_mask(mask_input):
    if isinstance(mask_input, torch.Tensor):
        mask_np = mask_input.cpu().numpy()
    elif isinstance(mask_input, np.ndarray):
        mask_np = mask_input
    else:
        raise TypeError(f"Mask input type {type(mask_input)} not supported. Expected torch.Tensor or np.ndarray.")

    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np.squeeze(0)
    elif mask_np.ndim == 4 and mask_np.shape[0] == 1 and mask_np.shape[1] == 1:
        mask_np = mask_np.squeeze(0).squeeze(0)

    if mask_np.ndim == 3:
        if mask_np.shape[-1] == 1:
            mask_np = mask_np.squeeze(axis=-1)
        elif mask_np.shape[0] == 1:
             mask_np = mask_np.squeeze(0)
        elif mask_np.shape[-1] > 1:
            mask_np = mask_np[..., 0]

    if mask_np.ndim != 2:
        raise ValueError(f"Mask has unexpected shape {mask_np.shape} after initial processing. Expected 2D mask (H,W). Original input shape was {mask_input.shape if hasattr(mask_input, 'shape') else 'unknown'}.")

    if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
        if mask_np.min() >= 0.0 and mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
    elif mask_np.dtype != np.uint8:
        mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)

    _, mask_np_binarized = cv2.threshold(mask_np, 128, 255, cv2.THRESH_BINARY)
    return mask_np_binarized

def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]
    if mask.sum() < 10: 
        return 0,h-1,0,w-1 
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2) 

def expand_image_mask(image, mask, ratio=1.4):
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio)
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W -w) // 2)
    w2 = W -w - w1

    pad_param_image = ((h1,h2),(w1,w2),(0,0)) if image.ndim == 3 else ((h1,h2),(w1,w2))
    pad_param_mask = ((h1,h2),(w1,w2))
    
    image_padded = np.pad(image, pad_param_image, 'constant', constant_values=255 if image.ndim == 3 else 0) 
    mask_padded = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image_padded, mask_padded

def pad_to_square(image, pad_value = 255, random_padding = False): 
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random_padding:
        padd_1 = int(np.random.randint(0,padd+1)) 
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if len(image.shape) == 2:
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2))
        else:
            pad_param = ((padd_1, padd_2), (0, 0))
    elif len(image.shape) == 3:
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2), (0, 0))
        else:
            pad_param = ((padd_1, padd_2), (0, 0), (0, 0))
    else:
        raise ValueError("Image must be 2D or 3D.")

    image_padded = np.pad(image, pad_param, 'constant', constant_values=pad_value)
    return image_padded

def create_highlighted_mask(image_np, mask_np, alpha=0.5, gray_value=128):
    if mask_np.max() <= 1.0 and mask_np.dtype != np.uint8: 
        mask_np_uint8 = (mask_np * 255).astype(np.uint8)
    elif mask_np.dtype != np.uint8: 
        mask_np_uint8 = np.clip(mask_np,0,255).astype(np.uint8)
    else:
        mask_np_uint8 = mask_np

    if mask_np_uint8.max() > 1: 
        mask_bool = mask_np_uint8 > 128
    else: 
        mask_bool = mask_np_uint8 > 0.5

    image_float = image_np.astype(np.float32)
    gray_overlay = np.full_like(image_float, gray_value, dtype=np.float32)
    result = image_float.copy()

    result[mask_bool] = (1 - alpha) * image_float[mask_bool] + alpha * gray_overlay[mask_bool]
    return result.astype(np.uint8)

def adjust_to_multiple_of_val(value, multiple):
    if multiple <= 0: 
        return value
    if value == 0: 
        return 0
    remainder = value % multiple
    if remainder == 0:
        return value
    return value + (multiple - remainder)

def calculate_adjusted_dimension_and_offset(start_coord, current_length, total_img_length, multiple):
    if current_length == 0 or total_img_length == 0:
        return start_coord, 0

    target_length = adjust_to_multiple_of_val(current_length, multiple)

    if start_coord + target_length <= total_img_length:
        return start_coord, target_length

    new_start_coord_for_target_length = total_img_length - target_length
    if new_start_coord_for_target_length >= 0:
        return new_start_coord_for_target_length, target_length

    max_fit_multiple = math.floor(total_img_length / float(multiple)) * multiple
    
    if max_fit_multiple > 0:
        original_center = start_coord + current_length / 2.0
        new_center = max_fit_multiple / 2.0
        aligned_start_coord = round(original_center - new_center)
        aligned_start_coord = max(0, aligned_start_coord)
        aligned_start_coord = min(aligned_start_coord, total_img_length - max_fit_multiple)
        return int(aligned_start_coord), int(max_fit_multiple)
    else:
        final_length = min(current_length, total_img_length - start_coord)
        final_length = max(1 if current_length > 0 and total_img_length - start_coord > 0 else 0, final_length) 
        return start_coord, final_length

class FillProcessNoScaling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", ),
                "ref_image": ("IMAGE", ), 
                "source_mask": ("MASK", ),
                "ref_mask": ("MASK", ),   
                "iterations": ("INT", {"default": 2, "min": 0, "max": 100}),
                "expand_pixels": ("INT", {"default": 0, "min": 0, "max": 1024}), 
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "BOX", "IMAGE")
    RETURN_NAMES = ("image", "mask", "old_tar_image", "tar_box_yyxx_crop", "preview_image")
    FUNCTION = "process"

    def process(self, source_image, ref_image, source_mask, ref_mask, iterations, expand_pixels):
        source_image_np = (source_image[0].cpu().numpy() * 255).astype(np.uint8)
        source_mask_np = standardize_and_binarize_mask(source_mask)
        ref_image_np = (ref_image[0].cpu().numpy() * 255).astype(np.uint8)
        ref_mask_np = standardize_and_binarize_mask(ref_mask)

        img_h, img_w = source_image_np.shape[:2]

        if iterations > 0:
            kernel = np.ones((7, 7), np.uint8) 
            dilated_source_mask_np = cv2.dilate(source_mask_np, kernel, iterations=iterations)
        else:
            dilated_source_mask_np = source_mask_np.copy()

        y1_exp, y2_exp, x1_exp, x2_exp = get_bbox_from_mask(dilated_source_mask_np) 

        initial_x1 = max(0, x1_exp - expand_pixels)
        initial_y1 = max(0, y1_exp - expand_pixels)
        initial_x2 = min(img_w - 1, x2_exp + expand_pixels) 
        initial_y2 = min(img_h - 1, y2_exp + expand_pixels) 
        
        if initial_x1 >= initial_x2 or initial_y1 >= initial_y2:
            initial_x1, initial_y1, initial_x2, initial_y2 = x1_exp,y1_exp,x2_exp,y2_exp
            if initial_x1 >= initial_x2 or initial_y1 >= initial_y2: 
                initial_x1, initial_y1, initial_x2, initial_y2 = 0, 0, img_w - 1, img_h - 1

        initial_bbox_w = initial_x2 - initial_x1 + 1
        initial_bbox_h = initial_y2 - initial_y1 + 1
        
        final_x1, final_bbox_w = calculate_adjusted_dimension_and_offset(initial_x1, initial_bbox_w, img_w, 8)
        final_y1, final_bbox_h = calculate_adjusted_dimension_and_offset(initial_y1, initial_bbox_h, img_h, 8)
        
        if initial_bbox_w > 0 and final_bbox_w == 0 and (img_w - final_x1) > 0 :
            final_bbox_w = min(initial_bbox_w, img_w - final_x1) 
            final_bbox_w = max(1, final_bbox_w) 
        if initial_bbox_h > 0 and final_bbox_h == 0 and (img_h - final_y1) > 0:
            final_bbox_h = min(initial_bbox_h, img_h - final_y1)
            final_bbox_h = max(1, final_bbox_h) 
        
        if final_bbox_w <= 0 or final_bbox_h <= 0:
            cropped_source_image = np.full((8,8,3), 255, dtype=np.uint8) 
            cropped_source_mask = np.zeros((8,8), dtype=np.uint8) 
            final_x1, final_y1, final_bbox_w, final_bbox_h = 0,0,8,8 
        else:
            cropped_source_image = source_image_np[final_y1 : final_y1 + final_bbox_h, final_x1 : final_x1 + final_bbox_w, :]
            cropped_source_mask = dilated_source_mask_np[final_y1 : final_y1 + final_bbox_h, final_x1 : final_x1 + final_bbox_w]
        
        ref_box_yyxx = get_bbox_from_mask(ref_mask_np)
        ref_mask_0_1 = ref_mask_np // 255
        ref_mask_3_0_1 = np.stack([ref_mask_0_1]*3, axis=-1)
        masked_ref_image = ref_image_np * ref_mask_3_0_1 + np.ones_like(ref_image_np, dtype=np.uint8) * 255 * (1 - ref_mask_3_0_1)
        
        ry1, ry2, rx1, rx2 = ref_box_yyxx
        if not(ry1 >= ry2 or rx1 >= rx2):
            masked_ref_image_cropped = masked_ref_image[ry1:ry2+1, rx1:rx2+1, :]
            ref_mask_cropped_for_expand = ref_mask_np[ry1:ry2+1, rx1:rx2+1]
        else: 
            masked_ref_image_cropped = np.full((100,100,3), 255, dtype=np.uint8) 
            ref_mask_cropped_for_expand = np.zeros((100,100), dtype=np.uint8)

        expanded_ref_image, _ = expand_image_mask(masked_ref_image_cropped, ref_mask_cropped_for_expand, ratio=1.3)
        processed_ref_image_square = pad_to_square(expanded_ref_image, pad_value=255)

        target_ref_h = final_bbox_h 
        orig_ref_h, orig_ref_w = processed_ref_image_square.shape[:2]
        
        if orig_ref_h == 0 or target_ref_h == 0: 
            scaled_ref_image = np.full((target_ref_h if target_ref_h > 0 else 8, 100 if target_ref_h > 0 else 8, 3), 255, dtype=np.uint8)
            if target_ref_h == 0 and final_bbox_h > 0: target_ref_h = scaled_ref_image.shape[0] # Update if placeholder used
        else:
            scale_ratio = target_ref_h / orig_ref_h
            target_ref_w = round(orig_ref_w * scale_ratio)
            if target_ref_w <=0: target_ref_w = 1 
            scaled_ref_image = cv2.resize(processed_ref_image_square, (target_ref_w, target_ref_h), interpolation=cv2.INTER_AREA)

        diptych_image = np.concatenate([scaled_ref_image, cropped_source_image], axis=1)
        
        mask_for_ref_part = np.zeros((target_ref_h, scaled_ref_image.shape[1]), dtype=np.uint8)
        if cropped_source_mask.ndim != 2: cropped_source_mask = cropped_source_mask[:,:,0]
        diptych_mask_np = np.concatenate([mask_for_ref_part, cropped_source_mask], axis=1)
        
        diptych_mask_3ch_for_preview = np.stack([diptych_mask_np]*3, axis=-1)
        show_diptych_image_np = create_highlighted_mask(diptych_image, diptych_mask_3ch_for_preview) 

        diptych_image_tensor = torch.from_numpy(diptych_image.astype(np.float32) / 255.0).unsqueeze(0)
        diptych_mask_tensor = torch.from_numpy(diptych_mask_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(1)
        original_source_image_tensor = source_image.clone()
        show_diptych_image_tensor = torch.from_numpy(show_diptych_image_np.astype(np.float32) / 255.0).unsqueeze(0)

        tar_box_yyxx_crop_value = (final_y1, final_y1 + final_bbox_h - 1, final_x1, final_x1 + final_bbox_w - 1)

        return (diptych_image_tensor, diptych_mask_tensor, original_source_image_tensor, tar_box_yyxx_crop_value, show_diptych_image_tensor)

class CropBackNoScaling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_image": ("IMAGE",), 
                "old_tar_image": ("IMAGE",),
                "tar_box_yyxx_crop": ("BOX",), 
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_back"

    def crop_back(self, raw_image, old_tar_image, tar_box_yyxx_crop): 
        processed_np = (raw_image[0].cpu().numpy() * 255).astype(np.uint8)
        original_source_np = (old_tar_image[0].cpu().numpy() * 255).astype(np.uint8)

        # Unpack tar_box_yyxx_crop
        # It was (y1, y2_inclusive, x1, x2_inclusive) from FillProcessNoScaling
        # We need final_bbox_x, final_bbox_y, final_bbox_width, final_bbox_height
        crop_y1, crop_y2, crop_x1, crop_x2 = tar_box_yyxx_crop
        final_bbox_x = crop_x1
        final_bbox_y = crop_y1
        final_bbox_width = crop_x2 - crop_x1 + 1
        final_bbox_height = crop_y2 - crop_y1 + 1
        
        processed_h, processed_w = processed_np.shape[:2]
        crop_x_start_from_right = max(0, processed_w - final_bbox_width)
        crop_y_start_from_right = 0 
        
        slice_h_from_processed = min(processed_h, final_bbox_height)
        slice_w_from_processed = min(final_bbox_width, processed_w - crop_x_start_from_right)

        if slice_w_from_processed <= 0 or slice_h_from_processed <= 0:
            return (old_tar_image,)

        generated_part_raw = processed_np[
            crop_y_start_from_right : crop_y_start_from_right + slice_h_from_processed,
            crop_x_start_from_right : crop_x_start_from_right + slice_w_from_processed,
            :
        ]
        
        if generated_part_raw.shape[0] != final_bbox_height or generated_part_raw.shape[1] != final_bbox_width:
            if final_bbox_width > 0 and final_bbox_height > 0 : # only resize if target is valid
                generated_part_resized = cv2.resize(generated_part_raw, (final_bbox_width, final_bbox_height), interpolation=cv2.INTER_AREA)
            else: # if target is 0, cannot resize to it, use raw (which might be empty)
                generated_part_resized = generated_part_raw
        else:
            generated_part_resized = generated_part_raw 
        
        output_image_np = original_source_np.copy()
        
        paste_y1 = final_bbox_y
        paste_x1 = final_bbox_x
        
        actual_paste_y1 = max(0, paste_y1)
        actual_paste_x1 = max(0, paste_x1)
        actual_paste_y2 = min(output_image_np.shape[0], paste_y1 + final_bbox_height)
        actual_paste_x2 = min(output_image_np.shape[1], paste_x1 + final_bbox_width)

        h_to_paste_in_original = actual_paste_y2 - actual_paste_y1
        w_to_paste_in_original = actual_paste_x2 - actual_paste_x1

        if h_to_paste_in_original > 0 and w_to_paste_in_original > 0 and \
           generated_part_resized.shape[0] >= h_to_paste_in_original and \
           generated_part_resized.shape[1] >= w_to_paste_in_original:
            
            slice_from_generated_y = slice(0, h_to_paste_in_original)
            slice_from_generated_x = slice(0, w_to_paste_in_original)
            
            target_paste_slice_y = slice(actual_paste_y1, actual_paste_y2)
            target_paste_slice_x = slice(actual_paste_x1, actual_paste_x2)

            try:
                output_image_np[target_paste_slice_y, target_paste_slice_x, :] = \
                    generated_part_resized[slice_from_generated_y, slice_from_generated_x, :]
            except IndexError:
                return (old_tar_image,) 
        else:
             pass # Skip paste if no valid area or source data too small

        output_image_tensor = torch.from_numpy(output_image_np.astype(np.float32) / 255.0).unsqueeze(0)
        return (output_image_tensor,)
