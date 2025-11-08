from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gc # <-- ADD THIS

# --- Model Loading (Lazy) ---
PROCESSOR = None
MODEL = None

def load_caption_model():
    """Loads the BLIP model into memory."""
    global PROCESSOR, MODEL
    if PROCESSOR is None or MODEL is None:
        try:
            print("LAZY LOADING: Loading BLIP model (this may take a moment)...")
            PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            print("BLIP model loaded successfully.")
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
            raise

# --- (generate_caption function stays the same) ---
def generate_caption(image: Image.Image) -> str:
    load_caption_model()
    if MODEL is None or PROCESSOR is None:
        return "Error: Captioning model not loaded."
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    try:
        inputs = PROCESSOR(image, return_tensors="pt")
        output_ids = MODEL.generate(**inputs, max_length=50)
        caption = PROCESSOR.decode(output_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error during caption generation: {e}")
        return "Error generating caption."

# --- ADD THIS NEW FUNCTION ---
def del_caption_model():
    """Deletes the BLIP model from memory."""
    global PROCESSOR, MODEL
    if PROCESSOR is not None:
        del PROCESSOR
        PROCESSOR = None
    if MODEL is not None:
        del MODEL
        MODEL = None
    gc.collect()
    print("Unloaded BLIP model from RAM.")