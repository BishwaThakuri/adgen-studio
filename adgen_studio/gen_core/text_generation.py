import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import gc # <-- ADD THIS

# --- Model Loading (Lazy) ---
TEXT_GENERATOR = None

def load_text_gen_model():
    """Loads the Gemma-2b-it model into memory."""
    global TEXT_GENERATOR
    if TEXT_GENERATOR is None:
        try:
            print("LAZY LOADING: Loading LLM (Gemma-2b-it)...")
            MODEL_NAME = "google/gemma-2b-it"
            TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
            MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
            ).to("cpu")
            TEXT_GENERATOR = pipeline(
                "text-generation", model=MODEL, tokenizer=TOKENIZER,
            )
            print("Gemma-2b-it model loaded successfully.")
        except Exception as e:
            print(f"Error loading Gemma model: {e}")
            raise

# --- (create_marketing_prompt function stays the same) ---
def create_marketing_prompt(caption: str) -> str:
    prompt = f"""<start_of_turn>user
You are an expert e-commerce copywriter.
Your task is to write 3 compelling, short ad variations for a product.
The product is: "{caption}".
Do not use hashtags. Keep the ads between 1-3 sentences.
Return ONLY the 3 ad variations, each starting with "Variation:".
<end_of_turn>
<start_of_turn>model
"""
    return prompt

# --- (generate_ad_copy function stays the same) ---
def generate_ad_copy(caption: str) -> list[str]:
    load_text_gen_model()
    if TEXT_GENERATOR is None:
        raise RuntimeError("Text generation model is not loaded.")
    prompt = create_marketing_prompt(caption)
    generation_args = {
        "max_new_tokens": 200, "return_full_text": False, "do_sample": True,
        "temperature": 0.7, "top_k": 50, "top_p": 0.95,
    }
    print(f"Generating ad copy for caption: '{caption}'")
    try:
        outputs = TEXT_GENERATOR(prompt, **generation_args)
        raw_text = outputs[0]['generated_text']
        variations = []
        parts = re.split(r"Variation.*:", raw_text)
        for part in parts:
            cleaned_line = part.replace('**', '').replace('\n', ' ').strip()
            if cleaned_line:
                variations.append(cleaned_line)
        return variations
    except Exception as e:
        print(f"Error during text generation: {e}")
        return ["Error generating ad copy."]

# --- ADD THIS NEW FUNCTION ---
def del_text_gen_model():
    """Deletes the Gemma model from memory."""
    global TEXT_GENERATOR
    if TEXT_GENERATOR is not None:
        del TEXT_GENERATOR
        TEXT_GENERATOR = None
    gc.collect()
    print("Unloaded Gemma model from RAM.")