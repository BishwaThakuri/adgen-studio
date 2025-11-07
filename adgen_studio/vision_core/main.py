from PIL import Image
import os
import tempfile

# Import our custom functions from the other files in this module
from .segmentation import remove_background
from .captioning import generate_caption

def process_image(input_image_path: str) -> dict:
    """
    Runs the full Sprint 1 vision pipeline on a single image.
    
    1. Opens the image.
    2. Generates a descriptive caption.
    3. Removes the background.
    4. Saves the segmented image to a temporary file.
    
    Args:
        input_image_path: The file path to the original product image.

    Returns:
        A dictionary containing the caption and the path to the
        newly created background-free image.
    """
    print(f"Starting full vision pipeline for: {input_image_path}")
    
    try:
        # Open the original image
        original_image = Image.open(input_image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return {"error": "Failed to open image."}

    # --- 1. Generate Caption ---
    # We use the original image for captioning to get more context
    print("Generating caption...")
    try:
        caption = generate_caption(original_image)
        print(f"Caption: {caption}")
    except Exception as e:
        print(f"Captioning failed: {e}")
        caption = "Error generating caption."

    # --- 2. Remove Background ---
    # We use the original image for segmentation
    print("Removing background...")
    try:
        segmented_image = remove_background(original_image)
        print("Background removal complete.")
    except Exception as e:
        print(f"Background removal failed: {e}")
        return {"error": "Failed to remove background."}

    # --- 3. Save Segmented Image ---
    # Create a temporary file to save the output PNG
    # We use a temp directory to keep our project folder clean
    try:
        temp_dir = tempfile.gettempdir()
        segmented_image_name = f"segmented_{os.path.basename(input_image_path)}.png"
        segmented_image_path = os.path.join(temp_dir, segmented_image_name)
        
        segmented_image.save(segmented_image_path, format="PNG")
        print(f"Segmented image saved to: {segmented_image_path}")
    except Exception as e:
        print(f"Failed to save segmented image: {e}")
        return {"error": "Failed to save segmented image."}

    # --- 4. Return Results ---
    return {
        "caption": caption,
        "segmented_image_path": segmented_image_path
    }