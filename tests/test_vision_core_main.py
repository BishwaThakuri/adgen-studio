from adgen_studio.vision_core.main import process_image
import os
import pprint

# Define the project root path to find the image
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("--- STARTING SPRINT 1 FINAL TEST ---")

# 1. Find the test image
input_path = os.path.join(PROJECT_ROOT, 'test-product.jpg')
if not os.path.exists(input_path):
    print(f"Error: test-product.jpg not found at {input_path}")
    exit()

# 2. Run the main processing function
results = process_image(input_path)

# 3. Print the results
print("\n--- SPRINT 1 FINAL RESULTS ---")
pprint.pprint(results)
print("----------------------------------")

if "error" not in results:
    print("Sprint 1 pipeline test SUCCESSFUL!")
    print(f"Check your temporary folder for the output image: {results['segmented_image_path']}")
else:
    print("Sprint 1 pipeline test FAILED.")