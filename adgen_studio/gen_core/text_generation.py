import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Model Loading ---
try:
    print("Loading LIGHTER LLM (Gemma-2b-it) for text generation...")
    
    # This is Google's 2-billion parameter model.
    MODEL_NAME = "google/gemma-2b-it"
    
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32, # Use float32 for CPU
    ).to("cpu") # Explicitly send to CPU

    TEXT_GENERATOR = pipeline(
        "text-generation",
        model=MODEL,
        tokenizer=TOKENIZER,
    )
    print("Gemma-2b-it model loaded successfully.")

except Exception as e:
    print(f"Error loading Gemma model: {e}")
    TEXT_GENERATOR = None
# ---------------------

def create_marketing_prompt(caption: str) -> str:
    """Creates a detailed instruction prompt for the Gemma model."""
    
    # This is the prompt format for Gemma.
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

def generate_ad_copy(caption: str) -> list[str]:
    """
    Generates compelling ad copy based on a simple product caption.
    """
    if TEXT_GENERATOR is None:
        raise RuntimeError("Text generation model is not loaded.")

    prompt = create_marketing_prompt(caption)

    generation_args = {
        "max_new_tokens": 200,
        "return_full_text": False,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
    }

    print(f"Generating ad copy for caption: '{caption}'")
    
    try:
        outputs = TEXT_GENERATOR(prompt, **generation_args)
        raw_text = outputs[0]['generated_text']

        # Clean the output
        variations = []
        # Split by the word "Variation" followed by any characters until a colon
        # This handles "Variation 1:", "**Variation 1:**", etc.
        import re
        parts = re.split(r"Variation.*:", raw_text)
        
        for part in parts:
            # Remove any remaining asterisks, newlines, and strip whitespace
            cleaned_line = part.replace('**', '').replace('\n', ' ').strip()
            if cleaned_line: # Make sure it's not an empty string
                variations.append(cleaned_line)
        
        return variations

    except Exception as e:
        print(f"Error during text generation: {e}")
        return ["Error generating ad copy."]