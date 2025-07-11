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

# where debug images live
DEBUG_FOLDER = "/tmp/debug"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# allow cross-origin calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalize_box(box, w, h):
    x, y, w, h = box
    return {
        "left": x / w,
        "top": y / h,
        "right": (x + w) / w,
        "bottom": (y + h) / h
    }

@app.get("/debug/{filename}")
def serve_debug_image(filename: str):
    path = os.path.join(DEBUG_FOLDER, filename)
    return FileResponse(path)

@app.post("/detect")
async def detect_card(data: dict):
    url = data.get("image_url")
    if not url:
        return {"error": "No image_url provided."}

    # fetch with a real browser UA
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = httpx.get(url, headers=headers, timeout=10.0)
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to download image: {e}"}

    # load into OpenCV
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    frame = np.array(img)
    height, width = frame.shape[:2]

    # 1) grayscale → adaptive threshold → clean up
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bw  = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2) find contours
    cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"error": "No contours found."}

    # 3) look for largest 4-point polygon
    quad = None
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4,2)
            break

    # 4) fallback: minAreaRect of the single largest contour
    if quad is None:
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        quad = cv2.boxPoints(rect).astype(int)

    # compute bounding box
    xs = quad[:,0]; ys = quad[:,1]
    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max()-xs.min()), int(ys.max()-ys.min())

    # draw debug image
    debug = frame.copy()
    cv2.drawContours(debug, [quad], -1, (0,0,255), 3)        # red polygon
    cv2.rectangle(debug, (x,y), (x+w,y+h), (0,255,0), 2)     # green AABB

    # save debug
    name = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(DEBUG_FOLDER, name)
    cv2.imwrite(path, cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))

    # normalized coords
    outer = normalize_box((x,y,w,h), width, height)
    inner = normalize_box((x + w*0.05, y + h*0.05, w*0.9, h*0.9), width, height)

    return {
        "outerEdges": outer,
        "innerEdges": inner,
        "rotation": float(cv2.minAreaRect(quad)[-1]),
        "confidence": 0.98,
        "debug_image_url": f"/debug/{name}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
