import torch
from PIL import Image
import numpy as np
import cv2
from torchvision.ops import box_convert

from groundingdino.util.inference import predict

from .config import cfg


__all__ = [
           "inference_groundingdino",
           "inference_sam",
           "inference_pix2pix",
           "apply_mask",
           "i2i_with_mask",
          ]


def inference_groundingdino(groundingdino_model, image: Image) -> np.ndarray:
    image_transformed, _ = cfg.transform(image, None)

    boxes, _, _ = predict(
        model=groundingdino_model, 
        image=image_transformed, 
        caption="person",
        box_threshold=0.3,
        text_threshold=0.2,
    )

    w, h = image.size
    boxes = box_convert(boxes=boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return boxes


def inference_sam(sam_model, image: Image, boxes: np.ndarray) -> np.ndarray:
    sam_model.set_image(np.array(image))

    masks, _, _ = sam_model.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
    )
    return masks


def inference_pix2pix(pix2pix_model, image: Image, prompt: str) -> Image:
    image_result = pix2pix_model(prompt, image=image.resize((512, 512)), 
                                 num_inference_steps=10, 
                                 image_guidance_scale=1).images[0]

    image_result = image_result.resize(image.size)
    return image_result


def apply_mask(image: Image, image_result: Image, masks: np.ndarray,
               mask_blur_kernel_size: int, mask_weight: float, mask_background: bool):
    image_np = np.array(image)
    image_result_np = np.array(image_result)
    for mask in masks:
        mask = np.copy(mask)
        mask = cv2.blur(mask.astype(np.float32), ksize=(mask_blur_kernel_size,mask_blur_kernel_size))
        mask = mask * mask_weight
        if mask_background:
            mask = 1 - mask
        image_np = image_np * mask[:,:,None] + image_result_np * (1-mask)[:,:,None]

    masked_result = Image.fromarray(image_np.astype(np.uint8))
    return masked_result


def i2i_with_mask(
        groundingdino_model, 
        sam_model, 
        pix2pix_model, 
        image: Image,
        prompt: str,
        mask_blur_kernel_size: int = 5,
        mask_weight: float = 1.0,
        mask_background: bool = False,
        ) -> Image:
    boxes = inference_groundingdino(groundingdino_model, image)
    masks = inference_sam(sam_model, image, boxes)
    image_result = inference_pix2pix(pix2pix_model, image, prompt)    
    masked_result = apply_mask(image, image_result, masks, 
                               mask_blur_kernel_size, mask_weight, mask_background)
    return masked_result
