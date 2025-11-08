import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
import os
import tempfile
import shutil
from PIL import Image
import gc # <-- IMPORT GC

# --- Import Your AI Modules ---
# Import the main functions AND the new delete functions
from adgen_studio.vision_core.main import process_image
from adgen_studio.vision_core.captioning import del_caption_model # <-- IMPORT DEL
from adgen_studio.gen_core.image_generation import create_mask_and_image, generate_new_image, del_inpainting_model # <-- IMPORT DEL
from adgen_studio.gen_core.text_generation import generate_ad_copy, del_text_gen_model # <-- IMPORT DEL

# --- Initialize FastAPI App ---
app = FastAPI(title="AdGen Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a Base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# --- Main API Endpoint ---
@app.post("/generate-ad-package/")
async def generate_ad_package(
    prompt: str = Form(...), 
    image: UploadFile = File(...)
):
    """
    Main endpoint to generate a full ad package.
    Receives a product image and a scene prompt.
    Returns ad copy and a new generated image.
    """
    
    temp_dir = tempfile.mkdtemp()
    temp_image_path = os.path.join(temp_dir, image.filename)

    try:
        # 1. Save uploaded image temporarily
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        print(f"--- API Call Started ---")
        print(f"Prompt: {prompt}")

        # --- Run Sprint 1: Vision Core ---
        print("Running Sprint 1: Vision Core...")
        sprint1_results = process_image(temp_image_path)
        if "error" in sprint1_results:
            raise HTTPException(status_code=500, detail=sprint1_results["error"])
        
        caption = sprint1_results["caption"]
        segmented_image_path = sprint1_results["segmented_image_path"]
        print(f"Sprint 1 Success. Caption: {caption}")
        
        # --- UNLOAD SPRINT 1 MODEL ---
        del_caption_model()
        gc.collect()

        # --- Run Sprint 2: Generation Core ---
        
        # 2a. Image Generation
        print("Running Sprint 2: Image Generation...")
        segmented_image = Image.open(segmented_image_path)
        base_image, mask_image = create_mask_and_image(segmented_image)
        generated_image = generate_new_image(base_image, mask_image, prompt)
        print("Sprint 2 Image Gen Success.")
        
        # --- UNLOAD SPRINT 2 (IMAGE) MODEL ---
        del_inpainting_model()
        del segmented_image, base_image, mask_image # Clear intermediate images
        gc.collect()
        
        # 2b. Text Generation
        print("Running Sprint 2: Text Generation...")
        ad_copy = generate_ad_copy(caption)
        print("Sprint 2 Text Gen Success.")
        
        # --- UNLOAD SPRINT 2 (TEXT) MODEL ---
        del_text_gen_model()
        gc.collect()

        # --- Prepare Response ---
        generated_image_b64 = image_to_base64(generated_image)
        print("--- API Call Successful ---")
        
        return JSONResponse(content={
            "ad_copy": ad_copy,
            "generated_image_b64": generated_image_b64
        })

    except Exception as e:
        print(f"--- API Call FAILED ---")
        print(f"Error: {e}")
        return HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up models (as a safety net) and temp files
        del_caption_model()
        del_inpainting_model()
        del_text_gen_model()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temp directory: {temp_dir}")
        gc.collect()

# --- Run the Server ---
if __name__ == "__main__":
    print("Starting AdGen Studio API server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)