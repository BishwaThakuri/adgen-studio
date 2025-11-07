import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageOps
import numpy as np

# --- Model Loading ---
try:
    print("Loading BETTER QUALITY Stable Diffusion 1.5 Inpainting model (512x512)...")
    
    # This is the 1.5-based inpainting model, known for more realistic results.
    PIPE = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32  # Use float32 for CPU
    ).to("cpu") # Explicitly send to CPU
    
    print("Stable Diffusion 1.5 model loaded successfully.")

except Exception as e:
    print(f"Error loading Stable Diffusion model: {e}")
    PIPE = None
# ---------------------


def create_mask_and_image(segmented_image: Image.Image, size=(512, 512)) -> tuple:
    """
    Prepares the segmented image and its mask for the inpainting model.
    (This function remains unchanged)
    """
    # Ensure image is RGBA to get the alpha channel
    if segmented_image.mode != "RGBA":
        segmented_image = segmented_image.convert("RGBA")

    # Resize with padding to maintain aspect ratio
    resized_image = ImageOps.pad(segmented_image, size, color=(0, 0, 0, 0))

    # Create the mask: White where the product is, Black everywhere else
    mask = resized_image.getchannel("A")
    mask = Image.fromarray(np.uint8(np.array(mask) > 128) * 255, 'L')
    
    # Create a solid white background
    base_image = Image.new("RGB", size, (255, 255, 255))
    
    # Paste the product (using its own alpha as the mask) onto the white bg
    base_image.paste(resized_image, (0, 0), resized_image)
    
    # Invert the mask for inpainting: White is "paint here", Black is "keep"
    inverted_mask = ImageOps.invert(mask)

    return base_image, inverted_mask


def generate_new_image(base_image: Image.Image, mask_image: Image.Image, prompt: str) -> Image.Image:
    """
    Generates a new image by inpainting a background based on a prompt.
    """
    if PIPE is None:
        raise RuntimeError("Stable Diffusion model is not loaded.")

    print(f"Generating image for prompt: '{prompt}'")
    
    negative_prompt = "low quality, blurry, deformed, disfigured, poor, repetitive, bad, ugly, lowres"

    # Generate the image
    generated_image = PIPE(
        prompt=prompt,
        image=base_image,
        mask_image=mask_image,
        negative_prompt=negative_prompt,
        num_inference_steps=25,  # Increased steps for better quality
        strength=1.0,
        guidance_scale=7.5,
    ).images[0]

    return generated_image