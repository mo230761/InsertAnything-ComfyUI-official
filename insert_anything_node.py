import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
import math
import torch
import numpy as np
import cv2
from PIL import Image


def standardize_and_binarize_mask(mask_input):
    """
    Converts a ComfyUI MASK input (torch.Tensor) to a standardized
    2D NumPy array (H, W), uint8, binarized (0 or 255).
    """
    if isinstance(mask_input, torch.Tensor):
        mask_np = mask_input.cpu().numpy()
    elif isinstance(mask_input, np.ndarray):
        mask_np = mask_input
    else:
        raise TypeError(f"Mask input type {type(mask_input)} not supported. Expected torch.Tensor or np.ndarray.")

    # Handle common ComfyUI mask shapes like (N, H, W) or (N, C, H, W)
    if mask_np.ndim == 3 and mask_np.shape[0] == 1:  # (1, H, W)
        mask_np = mask_np.squeeze(0)
    elif mask_np.ndim == 4 and mask_np.shape[0] == 1 and mask_np.shape[1] == 1: # (1,1,H,W)
        mask_np = mask_np.squeeze(0).squeeze(0)


    # Ensure single channel (H, W)
    if mask_np.ndim == 3:
        if mask_np.shape[-1] == 1:  # (H, W, 1)
            mask_np = mask_np.squeeze(axis=-1)
        elif mask_np.shape[0] == 1: # Also (1, H, W) if not caught above
             mask_np = mask_np.squeeze(0)
        elif mask_np.shape[-1] > 1: # (H, W, C) with C > 1, e.g., RGB mask passed incorrectly

            mask_np = mask_np[..., 0] 



    if mask_np.ndim != 2:
        raise ValueError(f"Mask has unexpected shape {mask_np.shape} after initial processing. Expected 2D mask (H,W). Original input shape was {mask_input.shape if hasattr(mask_input, 'shape') else 'unknown'}.")

    # Normalize to 0-255 uint8 if it's a float mask (0.0-1.0)
    if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
        if mask_np.min() >= 0.0 and mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)
        else: # Floats not in 0-1 range, clip and convert
            mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
    elif mask_np.dtype != np.uint8: # Other types, attempt to convert
        mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)

    # Binarize the mask: values > 128 become 255, others 0.
    _, mask_np_binarized = cv2.threshold(mask_np, 128, 255, cv2.THRESH_BINARY)
    return mask_np_binarized


def create_highlighted_mask(image_np, mask_np, alpha=0.5, gray_value=128):


    if mask_np.max() <= 1.0:
        mask_np = (mask_np * 255).astype(np.uint8)
    mask_bool = mask_np > 128

    image_float = image_np.astype(np.float32)

    gray_overlay = np.full_like(image_float, gray_value, dtype=np.float32)

    result = image_float.copy()
    result[mask_bool] = (
        (1 - alpha) * image_float[mask_bool] + alpha * gray_overlay[mask_bool]
    )

    return result.astype(np.uint8)

def f(r, T=0.6, beta=0.1):
    return np.where(r < T, beta + (1 - beta) / T * r, 1)

# Get the bounding box of the mask
def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)

# Expand the bounding box
def expand_bbox(mask, yyxx, ratio, min_crop=0):
    y1,y2,x1,x2 = yyxx
    H,W = mask.shape[0], mask.shape[1]

    yyxx_area = (y2-y1+1) * (x2-x1+1)
    r1 = yyxx_area / (H * W)
    r2 = f(r1)
    ratio = math.sqrt(r2 / r1)

    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)

# Pad the image to a square shape
def pad_to_square(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
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

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)

    return image

# Expand the image and mask
def expand_image_mask(image, mask, ratio=1.4):
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio) 
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W -w) // 2)
    w2 = W -w - w1

    pad_param_image = ((h1,h2),(w1,w2),(0,0))
    pad_param_mask = ((h1,h2),(w1,w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask

# Convert the bounding box to a square shape
def box2squre(image, box):
    H,W = image.shape[0], image.shape[1]
    y1,y2,x1,x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h,w = y2-y1, x2-x1

    if h >= w:
        x1 = cx - h//2
        x2 = cx + h//2
    else:
        y1 = cy - w//2
        y2 = cy + w//2
    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)

# Crop the predicted image back to the original image
def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 2 # maigin_pixel

    if W1 == H1:
        if m != 0:
            tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        else:
            tar_image[y1 :y2, x1:x2, :] =  pred[:, :]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    if m != 0:
        gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    else:
        gen_image[y1 :y2, x1:x2, :] =  pred[:, :]

    return gen_image


class MaskOption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "sketch": ("MASK", ),
                "upload": ("MASK", ),
                "option": (["sketch", "upload"], {"default": "sketch"}),

            }
        }

    RETURN_TYPES = ("MASK", )
    FUNCTION = "MaskOption"

    def MaskOption(self, sketch, upload, option):

        if option == "sketch":
            mask = sketch
        elif option == "upload":
            mask = upload

        return mask



class ReduxProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE", ),
                "ref_mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "ReduxProcess"

    def ReduxProcess(self, ref_image, ref_mask):

        ref_image_np = np.array(ref_image)[0] # Assuming ref_image is a batch of 1
        ref_image_np = (ref_image_np * 255).round().astype(np.uint8)

        # Standardize and binarize the reference mask
        ref_mask_np = standardize_and_binarize_mask(ref_mask) # ref_mask is the MASK input

        # Remove the background information of the reference picture
        # Ensure ref_mask_np is 2D before passing to get_bbox_from_mask
        ref_box_yyxx = get_bbox_from_mask(ref_mask_np)

        # Create a 0/1 mask for multiplication (0 for background, 1 for foreground)
        ref_mask_0_1 = ref_mask_np // 255 
        ref_mask_3_0_1 = np.stack([ref_mask_0_1, ref_mask_0_1, ref_mask_0_1], axis=-1)

        # Apply mask: foreground where mask is 1, white background where mask is 0
        masked_ref_image = ref_image_np * ref_mask_3_0_1 + \
                           np.ones_like(ref_image_np, dtype=np.uint8) * 255 * (1 - ref_mask_3_0_1)
        masked_ref_image = masked_ref_image.astype(np.uint8)

        # Extract the box where the reference image is located, and place the reference object at the center of the image
        y1,y2,x1,x2 = ref_box_yyxx
        # Ensure coordinates are valid for slicing
        y1, y2 = max(0, y1), min(masked_ref_image.shape[0], y2)
        x1, x2 = max(0, x1), min(masked_ref_image.shape[1], x2)
        
        if y1 >= y2 or x1 >= x2: # Handle empty crop case

            pass

        masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
        ref_mask_processed = ref_mask_np[y1:y2, x1:x2] # Use the binarized mask for consistency
        ratio = 1.3
        masked_ref_image, ref_mask_expanded = expand_image_mask(masked_ref_image, ref_mask_processed, ratio=ratio)
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
        # ref_mask_expanded is the corresponding mask for the expanded image, if needed later.


        # Extract the features of the reference image
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (768, 768)).astype(np.uint8)

        masked_ref_image = torch.from_numpy(masked_ref_image).unsqueeze(0).float() / 255.0

        return (masked_ref_image, )



class FillProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", ),
                "ref_image": ("IMAGE", ),
                "source_mask": ("MASK", ),
                "ref_mask": ("MASK", ),
                "iterations": ("INT", {"default": 2}),

            }
        }
    RETURN_TYPES = (
        "IMAGE", "MASK", "IMAGE", "BOX", "CROP_PARAMS", "IMAGE", # Adjusted
    )
    RETURN_NAMES = ( # Adjusted
        "image", "mask", "old_tar_image", "tar_box_yyxx_crop", "crop_params", "preview_image",
    )
    FUNCTION = "FillProcess"

    def FillProcess(self, source_image, ref_image, source_mask, ref_mask, iterations):

        size = (768, 768)

        ref_image_np = np.array(ref_image)[0] # Assuming batch of 1
        tar_image_np = np.array(source_image)[0] # Assuming batch of 1

        ref_image_np = (ref_image_np * 255).round().astype(np.uint8)
        tar_image_np = (tar_image_np * 255).round().astype(np.uint8)

        # Standardize and binarize masks
        ref_mask_np = standardize_and_binarize_mask(ref_mask) # ref_mask is MASK input
        tar_mask_np = standardize_and_binarize_mask(source_mask) # source_mask is MASK input

        # --- Reference Image Processing (similar to ReduxProcess) ---
        ref_box_yyxx = get_bbox_from_mask(ref_mask_np)
        
        ref_mask_0_1 = ref_mask_np // 255
        ref_mask_3_0_1 = np.stack([ref_mask_0_1, ref_mask_0_1, ref_mask_0_1], axis=-1)
        
        masked_ref_image = ref_image_np * ref_mask_3_0_1 + \
                           np.ones_like(ref_image_np, dtype=np.uint8) * 255 * (1 - ref_mask_3_0_1)
        masked_ref_image = masked_ref_image.astype(np.uint8)

        y1_ref, y2_ref, x1_ref, x2_ref = ref_box_yyxx
        y1_ref, y2_ref = max(0, y1_ref), min(masked_ref_image.shape[0], y2_ref)
        x1_ref, x2_ref = max(0, x1_ref), min(masked_ref_image.shape[1], x2_ref)

        if not(y1_ref >= y2_ref or x1_ref >= x2_ref): # Check for valid crop
            masked_ref_image = masked_ref_image[y1_ref:y2_ref, x1_ref:x2_ref, :]
            ref_mask_cropped = ref_mask_np[y1_ref:y2_ref, x1_ref:x2_ref]
        else: # Fallback if bbox is invalid
            ref_mask_cropped = ref_mask_np # use original mask if crop failed

        ratio_ref = 1.3
        masked_ref_image, _ = expand_image_mask(masked_ref_image, ref_mask_cropped, ratio=ratio_ref) # We mainly need the image
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
        # --- End Reference Image Processing ---

        # Dilate the target mask
        kernel = np.ones((7, 7), np.uint8)
        tar_mask_dilated = cv2.dilate(tar_mask_np, kernel, iterations=iterations)

        # Zoom in on the target image based on the dilated target mask
        tar_box_yyxx = get_bbox_from_mask(tar_mask_dilated)
        tar_box_yyxx_expanded = expand_bbox(tar_mask_dilated, tar_box_yyxx, ratio=1.2) # Pass tar_mask_dilated


        tar_box_yyxx_crop = expand_bbox(tar_image_np, tar_box_yyxx_expanded, ratio=2) 
        tar_box_yyxx_crop = expand_bbox(tar_mask_dilated, tar_box_yyxx_expanded, ratio=2)
        tar_box_yyxx_crop = box2squre(tar_image_np, tar_box_yyxx_crop) # box2square uses image for H,W
        
        y1_tar, y2_tar, x1_tar, x2_tar = tar_box_yyxx_crop

        old_tar_image_np = tar_image_np.copy()
        
        # Ensure crop coordinates are valid
        y1_tar, y2_tar = max(0, y1_tar), min(tar_image_np.shape[0], y2_tar)
        x1_tar, x2_tar = max(0, x1_tar), min(tar_image_np.shape[1], x2_tar)

        if y1_tar >= y2_tar or x1_tar >= x2_tar: # Handle empty crop
            # Fallback: use full image, or handle error
            # This indicates an issue with bbox calculation, possibly empty mask
            processed_tar_image = tar_image_np.copy()
            processed_tar_mask = tar_mask_dilated.copy() # or tar_mask_np
        else:
            processed_tar_image = tar_image_np[y1_tar:y2_tar, x1_tar:x2_tar, :]
            processed_tar_mask = tar_mask_dilated[y1_tar:y2_tar, x1_tar:x2_tar]


        H1, W1 = processed_tar_image.shape[0], processed_tar_image.shape[1]

        processed_tar_mask = pad_to_square(processed_tar_mask, pad_value=0)
        processed_tar_mask = cv2.resize(processed_tar_mask, size) # size is (768,768)

        # Resize reference image (already processed and padded to square)
        masked_ref_image_resized = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)

        processed_tar_image_padded = pad_to_square(processed_tar_image, pad_value=255)
        H2, W2 = processed_tar_image_padded.shape[0], processed_tar_image_padded.shape[1]

        processed_tar_image_resized = cv2.resize(processed_tar_image_padded, size)
        
        diptych_ref_tar = np.concatenate([masked_ref_image_resized, processed_tar_image_resized], axis=1)

        # Create diptych mask (mask is for the target part)
        processed_tar_mask_3ch = np.stack([processed_tar_mask, processed_tar_mask, processed_tar_mask], -1)
        
        mask_black_ref_part = np.zeros_like(masked_ref_image_resized, dtype=np.uint8) # Black mask for ref part
        mask_diptych = np.concatenate([mask_black_ref_part, processed_tar_mask_3ch], axis=1)

        show_diptych_ref_tar = create_highlighted_mask(diptych_ref_tar, mask_diptych) 


        output_mask_diptych_np = mask_diptych[:, :, 0] 

        diptych_ref_tar_tensor = torch.from_numpy(diptych_ref_tar.copy()).unsqueeze(0).float() / 255.0
        mask_diptych_tensor = torch.from_numpy(output_mask_diptych_np.copy()).unsqueeze(0).unsqueeze(1).float() / 255.0 # (B, C, H, W) so (1,1,H,W)

        mask_diptych_tensor = torch.from_numpy(output_mask_diptych_np.copy()).unsqueeze(0).float() / 255.0 # This was duplicated, using the (1,1,H,W) version above this line. Correcting.
        # The line above was: mask_diptych_tensor = torch.from_numpy(output_mask_diptych_np.copy()).unsqueeze(0).unsqueeze(1).float() / 255.0
        # The line below it was: mask_diptych_tensor = torch.from_numpy(output_mask_diptych_np.copy()).unsqueeze(0).float() / 255.0
        # The unsqueeze(1) is for channel, so (B, C, H, W) is (1,1,H,W) for a single channel mask.
        # Let's ensure the correct one is used. The one with unsqueeze(1) is typical for masks.
        # Re-evaluating: output_mask_diptych_np is (H,W).
        # For ComfyUI MASK type, it's often (B, H, W) or (B, 1, H, W).
        # If it's (B,H,W) then .unsqueeze(0).float() / 255.0 is fine.
        # If it needs to be (B,1,H,W) then .unsqueeze(0).unsqueeze(1).float() / 255.0 is correct.
        # The original code had both, the second one overwriting the first.
        # Let's stick to the (B, H, W) convention for mask output unless (B,1,H,W) is strictly needed by other nodes.
        # Given `mask_diptych_tensor = torch.from_numpy(output_mask_diptych_np.copy()).unsqueeze(0).float() / 255.0` was the last one, let's assume it's (B,H,W)
        # However, the type hint for MASK in ComfyUI is often a 2D tensor (H,W) or (1,H,W) for batch 1.
        # Let's assume the output MASK type should be (1, H, W) for consistency with typical ComfyUI mask tensors.
        # output_mask_diptych_np is (H,W). So, .unsqueeze(0) makes it (1,H,W). This is fine.

        old_tar_image_tensor = torch.from_numpy(old_tar_image_np.astype(np.float32) / 255.0).unsqueeze(0)
        crop_params_tuple = (H1, W1, H2, W2)
        # tar_box_yyxx_crop is already a tuple (y1,y2,x1,x2)

        show_diptych_ref_tar_tensor = torch.from_numpy(show_diptych_ref_tar.copy()).unsqueeze(0).float() / 255.0

        return (diptych_ref_tar_tensor, mask_diptych_tensor, old_tar_image_tensor, tar_box_yyxx_crop, crop_params_tuple, show_diptych_ref_tar_tensor)




class CropBack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_image": ("IMAGE",),
                "old_tar_image": ("IMAGE",), # Changed from custom type to IMAGE
                "tar_box_yyxx_crop": ("BOX",), # Assuming BOX is the type for the tuple
                "crop_params": ("CROP_PARAMS",) # Assuming CROP_PARAMS for (H1,W1,H2,W2)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "CropBack"

    def CropBack(
        self,
        raw_image,
        old_tar_image, # This is now a tensor
        tar_box_yyxx_crop, # This is a tuple (y1,y2,x1,x2)
        crop_params      # This is a tuple (H1,W1,H2,W2)
    ):
        # Unpack crop_params
        H1, W1, H2, W2 = crop_params

        # Convert old_tar_image tensor to numpy array for crop_back helper
        old_tar_image_np = (old_tar_image[0].cpu().numpy() * 255).round().astype(np.uint8)

        raw_image_np = np.array(raw_image)[0] # raw_image is already a tensor
        raw_image_np = (raw_image_np * 255).round().astype(np.uint8)
        # Convert raw_image_np to PIL Image for cropping the right half
        pil_raw_image = Image.fromarray(raw_image_np)

        width, height = pil_raw_image.size
        left = width // 2
        right = width
        top = 0
        bottom = height
        edited_image_pil = pil_raw_image.crop((left, top, right, bottom))

        edited_image_np = np.array(edited_image_pil)
        
        # tar_box_yyxx_crop is already a tuple, convert to numpy array if crop_back expects it
        # The crop_back helper function expects tar_image (old_tar_image_np) and tar_box_yyxx_crop as numpy arrays.
        # extra_sizes is also expected as a numpy array.
        edited_image_final_np = crop_back(edited_image_np, old_tar_image_np, 
                                          np.array([H1, W1, H2, W2]), 
                                          np.array(tar_box_yyxx_crop))

        edited_image_tensor = torch.from_numpy(edited_image_final_np.astype(np.float32) / 255.0).unsqueeze(0)

        return (edited_image_tensor,)
        raw_image = (raw_image * 255).round().astype(np.uint8)
        raw_image = Image.fromarray(raw_image)

        # This block is now part of the new structure above.
        # raw_image = np.array(raw_image)[0]
        # raw_image = (raw_image * 255).round().astype(np.uint8)
        # raw_image = Image.fromarray(raw_image)

        # width, height = raw_image.size
        # left = width // 2
        # right = width
        # top = 0
        # bottom = height
        # edited_image = raw_image.crop((left, top, right, bottom))

        # edited_image = np.array(edited_image)
        # edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop))

        # edited_image = torch.from_numpy(edited_image).unsqueeze(0).float() / 255.0

        # return (edited_image,)
        pass # Placeholder for the removed block, the logic is now in the new method body
