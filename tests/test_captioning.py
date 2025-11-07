from PIL import Image
from adgen_studio.vision_core.captioning import generate_caption
import os

# Define the project root path to find the image
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("Starting captioning test...")

# 1. Open the test image
try:
    input_path = os.path.join(PROJECT_ROOT, 'test-product.jpg')
    input_image = Image.open(input_path)
    print("Successfully loaded test-product.jpg")
except FileNotFoundError:
    print(f"Error: test-product.jpg not found at {input_path}")
    exit()

# 2. Run the function
print("Generating caption...")
caption = generate_caption(input_image)

# 3. Print the result
print("\n--- TEST RESULT ---")
print(f"Generated Caption: {caption}")
print("---------------------")