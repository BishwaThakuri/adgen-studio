import streamlit as st
import requests
from PIL import Image
import io
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="AdGen Studio",
    page_icon="✨",
    layout="wide",
)

# --- API Configuration ---
# This is the URL of your FastAPI backend.
# If you are running both on your local machine, this is correct.
BACKEND_URL = "http://127.0.0.1:8000/generate-ad-package/"

# --- Helper Function ---
def base64_to_image(b64_string: str) -> Image.Image:
    """Converts a Base64 string back to a PIL Image."""
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes))
    return img

# --- Main App Interface ---
st.title("✨ AdGen Studio")
st.header("AI-Powered Product Marketing Suite")
st.markdown("""
Upload a photo of your product, and our AI will generate new lifestyle images 
and professional ad copy in minutes.
""")

st.warning("**Important:** Your FastAPI backend server must be running in a separate terminal for this to work. Run `python main.py` in your project folder.")

st.divider()

# --- Input Form ---
with st.form(key="ad_form"):
    
    # Left Column: Image Upload
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Upload Your Product Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Right Column: Text Prompt
    with col2:
        st.subheader("2. Describe the New Scene")
        prompt = st.text_input(
            "Enter a prompt for the new image:",
            placeholder="e.g., 'on a white marble table, next to a small plant'"
        )

    # Submit Button
    st.divider()
    submit_button = st.form_submit_button("🚀 Generate My Ad Package!")

# --- Form Submission Logic ---
if submit_button:
    if uploaded_file is None:
        st.error("❌ Please upload an image.")
    elif not prompt:
        st.error("❌ Please enter a prompt for the scene.")
    else:
        with st.spinner("🚀 AI is warming up... This will take 4-6 minutes..."):
            try:
                # Prepare the files and data for the POST request
                files = {'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {'prompt': prompt}
                
                # Make the API call to your FastAPI backend
                response = requests.post(BACKEND_URL, files=files, data=data, timeout=600) # 10 min timeout
                
                if response.status_code == 200:
                    # Success!
                    result = response.json()
                    ad_copy = result.get("ad_copy", [])
                    img_b64 = result.get("generated_image_b64")

                    st.success("✅ Your ad package is ready!")
                    st.balloons()
                    
                    st.header("Generated Image")
                    gen_image = base64_to_image(img_b64)
                    st.image(gen_image, caption="AI-Generated Lifestyle Image", use_column_width=True)

                    st.header("Generated Ad Copy")
                    for i, ad in enumerate(ad_copy, 1):
                        st.markdown(f"**Variation {i}:**")
                        st.info(ad)

                else:
                    # Handle API errors
                    st.error(f"Error from API: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: Could not connect to the backend API. Is it running?")
                st.error(f"Details: {e}")