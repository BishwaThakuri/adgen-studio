from PIL import Image
from adgen_studio.vision_core.segmentation import remove_background
import os

# Define the project root path to find the image
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("Starting test...")

# 1. Open the test image
try:
    input_path = os.path.join(PROJECT_ROOT, 'test-product.jpg')
    input_image = Image.open(input_path)
    print("Successfully loaded test-product.jpg")
except FileNotFoundError:
    print(f"Error: test-product.jpg not found at {input_path}")
    print("Please add a test-product.jpg file to your project's root folder.")
    exit()

# 2. Run the function
print("Removing background...")
output_image = remove_background(input_image)
print("Background removal complete.")

# 3. Save the result
output_path = os.path.join(PROJECT_ROOT, 'test-product-output.png')
output_image.save(output_path)

print(f"Success! Output image saved to: {output_path}")