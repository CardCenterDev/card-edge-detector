from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import cv2
import numpy as np
import httpx
import io
import os
from PIL import Image
import uuid

app = FastAPI()

# Render’s writable directory
DEBUG_FOLDER = "/tmp/debug"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# Enable CORS for any origin (you can lock this down later)
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
async def serve_debug_image(filename: str):
    path = os.path.join(DEBUG_FOLDER, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "Debug image not found."}, status_code=404)
    return FileResponse(path)

@app.post("/process_image")
async def process_image(data: dict):
    image_url = data.get("imageUrl") or data.get("image_url")
    if not image_url:
        return JSONResponse({"error": "No imageUrl provided."}, status_code=400)

    # 1) download
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = httpx.get(image_url, headers=headers, timeout=15.0)
        resp.raise_for_status()
    except Exception as e:
        return JSONResponse({"error": f"Failed to download image: {e}"}, status_code=400)

    # 2) decode
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    image_np = np.array(img)
    h, w = image_np.shape[:2]

    # 3) preprocess + edges
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 4) find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return JSONResponse({"error": "No contours found."}, status_code=422)

    # 5) pick the biggest
    best = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < (w * h * 0.05):
        return JSONResponse({"error": "Largest contour too small."}, status_code=422)

    # 6) get bounding rect + minAreaRect
    x,y,ww,hh = cv2.boundingRect(best)
    rot = cv2.minAreaRect(best)
    box = cv2.boxPoints(rot)
    box = np.intp(box)

    # 7) draw debug overlay
    dbg = image_np.copy()
    cv2.drawContours(dbg, [box], 0, (0,0,255), 2)
    cv2.rectangle(dbg, (x,y), (x+ww,y+hh), (0,255,0), 2)

    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(DEBUG_FOLDER, filename)
    cv2.imwrite(path, cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))

    # 8) normalize
    outer = normalize_box((x,y,ww,hh), w, h)
    inner = normalize_box((x+ww*0.1, y+hh*0.1, ww*0.8, hh*0.8), w, h)

    # 9) full debug URL
    base = os.environ.get("RENDER_EXTERNAL_URL") or ""  # set via Render dashboard
    debug_url = f"{base}/debug/{filename}"

    return {
        "outerEdges": outer,
        "innerEdges": inner,
        "rotation": rot[-1],
        "confidence": 0.95,
        "debug_image_url": debug_url
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
