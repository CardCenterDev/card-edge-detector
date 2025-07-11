from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import cv2
import numpy as np
import httpx  # Replaces requests
import io
import os
from PIL import Image
import uuid

app = FastAPI()

# Use /tmp for Render's writable filesystem
DEBUG_FOLDER = "/tmp/debug"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# Enable CORS for frontend/backend communication
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

@app.get("/debug/{filename}")
def serve_debug_image(filename: str):
    path = os.path.join(DEBUG_FOLDER, filename)
    return FileResponse(path)

@app.post("/detect")
async def detect_card(data: dict):
    image_url = data.get("image_url")
    if not image_url:
        return {"error": "No image_url provided."}

    # Use httpx with user-agent to avoid 403/429 errors
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = httpx.get(image_url, headers=headers, timeout=10.0)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to download image: {str(e)}"}

    # Convert to OpenCV format
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    # Preprocess
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Detect contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "No contours found."}

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    rotated_rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rotated_rect)
    box = np.intp(box)

    # Draw debug image
    debug_image = image_np.copy()
    cv2.drawContours(debug_image, [box], 0, (0, 0, 255), 2)  # red
    x, y, w, h = cv2.boundingRect(largest)
    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green

    # Save debug image
    debug_filename = f"{uuid.uuid4().hex}.jpg"
    debug_path = os.path.join(DEBUG_FOLDER, debug_filename)
    cv2.imwrite(debug_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

    # Return normalized box and debug URL
    outer = normalize_box((x, y, w, h), width, height)
    inner = normalize_box((x + w * 0.1, y + h * 0.1, w * 0.8, h * 0.8), width, height)

    return {
        "outerEdges": outer,
        "innerEdges": inner,
        "rotation": rotated_rect[-1],
        "confidence": 0.9,
        "debug_image_url": f"/debug/{debug_filename}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
