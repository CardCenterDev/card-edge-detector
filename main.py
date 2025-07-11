from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import cv2
import numpy as np
import httpx
import io
import os
from PIL import Image
import uuid

app = FastAPI()

# Writable directory for Render
DEBUG_FOLDER = "/tmp/debug"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# Allow CORS
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

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = httpx.get(image_url, headers=headers, timeout=10.0)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to download image: {str(e)}"}

    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "No contours found."}

    # Filter contours that form a 4-sided polygon and have a card-like aspect ratio
    rect_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:  # Lower threshold
            continue

        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = max(w, h) / min(w, h)
            if 1.2 < aspect_ratio < 1.7:
                rect_candidates.append((area, c))

    # If no good rectangles found, fallback to largest contour
    if rect_candidates:
        _, best_contour = max(rect_candidates, key=lambda x: x[0])
    else:
        best_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_contour)
        if area < 500:
            return {"error": f"Largest contour too small to be a card. Area={area}"}

    rotated_rect = cv2.minAreaRect(best_contour)
    box = cv2.boxPoints(rotated_rect)
    box = np.intp(box)
    x, y, w, h = cv2.boundingRect(best_contour)

    # Draw debug image
    debug_image = image_np.copy()
    cv2.drawContours(debug_image, [box], 0, (0, 0, 255), 2)  # red rotated box
    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green bounding box

    debug_filename = f"{uuid.uuid4().hex}.jpg"
    debug_path = os.path.join(DEBUG_FOLDER, debug_filename)
    cv2.imwrite(debug_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

    # Return normalized coordinates and metadata
    outer = normalize_box((x, y, w, h), width, height)
    inner = normalize_box((x + w * 0.1, y + h * 0.1, w * 0.8, h * 0.8), width, height)

    return {
        "outerEdges": outer,
        "innerEdges": inner,
        "rotation": rotated_rect[-1],
        "confidence": 0.95,
        "debug_image_url": f"/debug/{debug_filename}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
