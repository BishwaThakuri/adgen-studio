import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageOps
import numpy as np
import gc # <-- ADD THIS

# --- Model Loading (Lazy) ---
PIPE = None

def load_inpainting_model():
    """Loads the Stable Diffusion 1.5 model into memory."""
    global PIPE
    if PIPE is None:
        try:
            print("LAZY LOADING: Loading BETTER QUALITY Stable Diffusion 1.5 Inpainting model...")
            PIPE = AutoPipelineForInpainting.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float32
            ).to("cpu")
            print("Stable Diffusion 1.5 model loaded successfully.")
        except Exception as e:
            print(f"Error loading Stable Diffusion model: {e}")
            raise

# --- (create_mask_and_image function stays the same) ---
def create_mask_and_image(segmented_image: Image.Image, size=(512, 512)) -> tuple:
    if segmented_image.mode != "RGBA":
        segmented_image = segmented_image.convert("RGBA")
    resized_image = ImageOps.pad(segmented_image, size, color=(0, 0, 0, 0))
    mask = resized_image.getchannel("A")
    mask = Image.fromarray(np.uint8(np.array(mask) > 128) * 255, 'L')
    base_image = Image.new("RGB", size, (255, 255, 255))
    base_image.paste(resized_image, (0, 0), resized_image)
    inverted_mask = ImageOps.invert(mask)
    return base_image, inverted_mask

# --- (generate_new_image function stays the same) ---
def generate_new_image(base_image: Image.Image, mask_image: Image.Image, prompt: str) -> Image.Image:
    load_inpainting_model()
    if PIPE is None:
        raise RuntimeError("Stable Diffusion model is not loaded.")
    print(f"Generating image for prompt: '{prompt}'")
    negative_prompt = "low quality, blurry, deformed, disfigured, poor, repetitive, bad, ugly, lowres"
    generated_image = PIPE(
        prompt=prompt, image=base_image, mask_image=mask_image,
        negative_prompt=negative_prompt, num_inference_steps=25,
        strength=1.0, guidance_scale=7.5,
    ).images[0]
    return generated_image

# --- ADD THIS NEW FUNCTION ---
def del_inpainting_model():
    """Deletes the Stable Diffusion model from memory."""
    global PIPE
    if PIPE is not None:
        del PIPE
        PIPE = None
    gc.collect()
    print("Unloaded Stable Diffusion model from RAM.")