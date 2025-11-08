from adgen_studio.gen_core.text_generation import generate_ad_copy
import pprint

# --- This is the caption we got from Sprint 1 ---
INPUT_CAPTION = "there is a white cup of coffee on a wooden table"
# ------------------------------------------------

print("--- STARTING SPRINT 2 (TEXT GEN) TEST ---")

# 1. Run the generation function
try:
    ad_variations = generate_ad_copy(INPUT_CAPTION)

    # 2. Print the results
    print("\n--- TEST RESULT ---")
    print(f"Input Caption: {INPUT_CAPTION}")
    print("\nGenerated Ad Copy:")
    pprint.pprint(ad_variations)
    print("---------------------")

except Exception as e:
    print(f"\nText generation failed: {e}")