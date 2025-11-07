from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# --- Model Loading ---
# We load the model and processor once when this module is imported.
# This avoids reloading the large model every time the function is called.
try:
    print("Loading BLIP model (this may take a moment)...")
    PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    print("BLIP model loaded successfully.")
except Exception as e:
    print(f"Error loading BLIP model: {e}")
    print("Please check your internet connection and transformers installation.")
    PROCESSOR = None
    MODEL = None
# ---------------------


def generate_caption(image: Image.Image) -> str:
    """
    Generates a descriptive caption for a given PIL Image using the BLIP model.

    Args:
        image: The input PIL Image object (in RGB mode).

    Returns:
        A string containing the generated caption.
    """
    if MODEL is None or PROCESSOR is None:
        return "Error: Captioning model not loaded."

    # Ensure image is in RGB format, as required by BLIP
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    
    try:
        # Process the image
        inputs = PROCESSOR(image, return_tensors="pt")

        # Generate the caption
        output_ids = MODEL.generate(**inputs, max_length=50)

        # Decode the caption
        caption = PROCESSOR.decode(output_ids[0], skip_special_tokens=True)
        
        return caption
    
    except Exception as e:
        print(f"Error during caption generation: {e}")
        return "Error generating caption."