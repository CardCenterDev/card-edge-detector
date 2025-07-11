# main.py
from fastapi import FastAPI, Request
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

# Writable debug folder on Render
DEBUG_FOLDER = "/tmp/debug"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

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
        "bottom": (y + h) / height,
    }

@app.get("/debug/{filename}")
def serve_debug_image(filename: str):
    path = os.path.join(DEBUG_FOLDER, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "debug image not found"}, status_code=404)
    return FileResponse(path)

@app.post("/process_image")
async def process_image(req: Request):
    body = await req.json()
    image_url = body.get("imageUrl") or body.get("image_url")
    if not image_url:
        return JSONResponse({"error": "No imageUrl provided"}, status_code=400)

    # 1) download with httpx
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = httpx.get(image_url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return JSONResponse({"error": f"Failed to download image: {e}"}, status_code=400)

    # 2) load into cv2
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # 3) grayscale + blur
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 4) adaptive threshold to isolate bright border
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,    # block size
        -10    # C value (tune down to pick up pink/white border)
    )
    # invert so card area is white
    thresh = cv2.bitwise_not(thresh)

    # 5) morphological close to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 6) find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return JSONResponse({"error":"No contours found."}, status_code=500)

    # 7) pick the largest by area
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < (w*h*0.1):
        return JSONResponse({"error":"Detected region too small."}, status_code=500)

    # 8) get rotated rect + bounding box
    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect).astype(int)
    x,y,box_w,box_h = cv2.boundingRect(best)

    # 9) save a debug overlay
    debug = img_np.copy()
    cv2.drawContours(debug, [box], -1, (0,0,255), 3)             # red rotated box
    cv2.rectangle(debug, (x,y),(x+box_w,y+box_h),(0,255,0),3)     # green AABB
    dbg_name = f"{uuid.uuid4().hex}.jpg"
    dbg_path = os.path.join(DEBUG_FOLDER, dbg_name)
    cv2.imwrite(dbg_path, cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))

    # 10) normalize and respond
    outer = normalize_box((x,y,box_w,box_h), w, h)
    # simple inset for inner border
    inner = normalize_box(
        (x+box_w*0.05, y+box_h*0.05, box_w*0.9, box_h*0.9),
        w,h
    )
    angle = rect[-1]
    # normalize angle to [-45,45]
    if angle < -45: angle += 90
    if angle > 45:  angle -= 90

    return {
        "outerEdges":  outer,
        "innerEdges":  inner,
        "rotation":    round(angle,1),
        "confidence":  0.95,
        "debug_image_url": f"/debug/{dbg_name}"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
