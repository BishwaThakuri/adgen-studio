from PIL import Image
from adgen_studio.gen_core.image_generation import create_mask_and_image, generate_new_image
import os

# --- PASTE YOUR PATH FROM SPRINT 1 HERE ---
SEGMENTED_IMAGE_PATH = r"C:\Users\thaku\AppData\Local\Temp\segmented_test-product.jpg.png"
# -------------------------------------------

# Define the project root path to save the image
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'test-product-inpainted.png')

print("--- STARTING SPRINT 2 (IMAGE GEN) TEST ---")

# 1. Open the segmented image
try:
    segmented_image = Image.open(SEGMENTED_IMAGE_PATH)
    print("Successfully loaded segmented image.")
except FileNotFoundError:
    print(f"Error: Segmented image not found at {SEGMENTED_IMAGE_PATH}")
    print("Please re-run the Sprint 1 test and paste the correct path.")
    exit()

# 2. Define our target scene
scene_prompt = "A white coffee cup, on a rustic wooden coffee shop table, morning light"

# 3. Prepare the image and mask for the model
print("Creating mask and base image...")
base_image, mask_image = create_mask_and_image(segmented_image)

# Save the mask for debugging (optional)
# mask_image.save(os.path.join(PROJECT_ROOT, 'test-mask.png'))
# base_image.save(os.path.join(PROJECT_ROOT, 'test-base.png'))

# 4. Run the generation function
print("Generating new image (this will take time)...")
try:
    final_image = generate_new_image(base_image, mask_image, scene_prompt)

    # 5. Save the final result
    final_image.save(OUTPUT_PATH)
    print("\n--- TEST RESULT ---")
    print(f"Success! New image for new model is saved to: {OUTPUT_PATH}")
    print("---------------------")

except Exception as e:
    print(f"\nImage generation failed: {e}")
    print("This often happens if you run out of memory (RAM or VRAM).") 