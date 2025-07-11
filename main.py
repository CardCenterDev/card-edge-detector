from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import requests
import io
from PIL import Image

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect_card(data: dict):
    image_url = data.get("image_url")
    if not image_url:
        return {"error": "No image_url provided."}

    # Download image
    response = requests.get(image_url)
    if response.status_code != 200:
        return {"error": "Failed to download image."}

    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    # === Simple placeholder logic ===
    outer = {"left": 0.05, "top": 0.05, "right": 0.95, "bottom": 0.95}
    inner = {"left": 0.10, "top": 0.10, "right": 0.90, "bottom": 0.90}
    return {
        "outerEdges": outer,
        "innerEdges": inner,
        "rotation": 0.0,
        "confidence": 0.95
    }

# Optional: only for local dev
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
