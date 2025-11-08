# ✨ AdGen Studio

**AdGen Studio** is a full-stack, AI-powered marketing suite that generates new lifestyle product images and compelling ad copy from a single product photo.

This project is an end-to-end demonstration of a modern, multi-model AI pipeline, from backend API development with FastAPI to a user-friendly frontend with Streamlit.

(Recommended: Record a GIF of your app working and place it here!)
`![AdGen Studio Demo GIF]`

## 🚀 The Problem

Small businesses and e-commerce entrepreneurs need high-quality, diverse marketing assets to stand out. However, hiring photographers for lifestyle shoots and professional copywriters is expensive and time-consuming.

AdGen Studio solves this by turning a single, basic product photo into a complete ad package in minutes, using the power of generative AI.

## 🛠️ Key Features

* **AI Image Analysis:** Uses a **BLIP** model to analyze the product and generate a descriptive caption.
* **Automated Background Removal:** Intelligently isolates the product from its original background.
* **Generative Image Inpainting:** Uses **Stable Diffusion (1.5)** to place the product into new, context-aware "lifestyle" scenes based on a text prompt.
* **Generative Ad Copy:** Uses **Gemma 2B** to write multiple creative, professional, and engaging ad copy variations based on the product's description.
* **Full-Stack Architecture:** Built with a decoupled **FastAPI** backend (for AI processing) and a **Streamlit** frontend (for user interaction).
* **Memory-Safe:** Implements an "aggressive lazy loading" pattern to manage memory, allowing this complex, multi-model pipeline to run on consumer hardware.

## 🧠 How It Works: The AI Pipeline

The application's backend runs a sequential, multi-model AI pipeline for every request.

1.  **Input:** The user uploads an image (`product.jpg`) and a text prompt (`"on a marble table"`).
2.  **Sprint 1: Vision Core (Analysis)**
    * The image is sent to the `vision_core` module.
    * **BLIP** loads, analyzes the image, and generates a base caption (e.g., `"a white coffee cup"`).
    * The BLIP model is **unloaded** from memory.
    * The background is removed, creating a `product.png` with a transparent background.
3.  **Sprint 2: Gen Core (Creation)**
    * **Image Gen:** The **Stable Diffusion 1.5 Inpainting** model is **loaded**. It uses the `product.png`, its mask, and the user's prompt to generate a new lifestyle image.
    * The Stable Diffusion model is **unloaded** from memory.
    * **Text Gen:** The **Gemma 2B** model is **loaded**. It takes the caption from Sprint 1 and generates 3 ad copy variations.
    * The Gemma model is **unloaded** from memory.
4.  **Output:** The FastAPI backend returns a JSON object to the Streamlit frontend, containing the new ad copy and the Base64-encoded generated image.

## 💻 Tech Stack

| Category | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Backend** | FastAPI, Uvicorn |
| **AI Frameworks**| Hugging Face Transformers, Hugging Face Diffusers, PyTorch |
| **Core Models** | `Salesforce/blip` (Captioning), `runwayml/stable-diffusion-inpBainting` (Image Gen), `google/gemma-2b-it` (Text Gen) |
| **Core Libraries**| Python 3.10, `rembg` (Background Removal), `requests`, `Pillow` |

## ⚙️ How to Run This Project Locally

This project requires a `conda` or `venv` environment and a Hugging Face account.

### Prerequisites
* Python 3.10+
* Git
* A Hugging Face account (for Gemma access)

### 1. Clone the Repository
```bash
git clone [https://github.com/BishwaThakuri/adgen-studio.git](https://github.com/BishwaThakuri/adgen-studio.git)
cd adgen-studio
```

### 2. Set Up the Python Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# Windows
.\venv\Scripts\Activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Log in to Hugging Face
You must be logged in to download the gated gemma model.
- Go to huggingface.co/google/gemma-2b-it and accept the terms.
- Go to your Hugging Face Settings > Access Tokens to create a "read" token.
- In your terminal, run the login command and paste your token:
```bash
huggingface-cli login
```

### 5. Run the Application
This application has two parts. You must run them in two separate terminals.
**Terminal 1: Run the Backend (FastAPI)**
```bash
python main.py
```
**Note:** The server will start instantly. The AI models will only load when you make the first API request. This first request will take 4-6 minutes.

**Terminal 2: Run the Frontend (Streamlit)**
```bash
streamlit run app.p
```
Your browser will automatically open to `http://localhost:8501`

### 6. Use the App
1. Make sure both servers are running.
2. Open `http://localhost:8501` in your browser.
3. Upload a product photo (like `test-product.jpg`).
4. Enter a scene prompt (e.g., "on a beach at sunset").
5. Click "Generate My Ad Package!" and wait 4-6 minutes for the pipeline to complete.
6. See your results!