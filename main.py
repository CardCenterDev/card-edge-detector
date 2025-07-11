from fastapi import FastAPI
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalize_box(box, width, height):
    x, y, w, h = box
    return {
        "left": x / width,
        "top": y / height,
        "right": (x + w) / width,
        "bottom": (y + h) / height
    }

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

    # Convert to grayscale and blur
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "No contours found."}

    # Find the largest contour (assumed to be the card)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    outer = normalize_box((x, y, w, h), width, height)

    # Optional: fake "inner" box for now (shrink outer by 10%)
    padding_w = w * 0.1
    padding_h = h * 0.1
    inner = normalize_box((x + padding_w, y + padding_h, w * 0.8, h * 0.8), width, height)

    return {
        "outerEdges": outer,
        "innerEdges": inner,
        "rotation": 0.0,
        "confidence": 0.8  # Just a placeholder for now
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
