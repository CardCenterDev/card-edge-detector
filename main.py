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

DEBUG_FOLDER = "/tmp/debug"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalize_box(box, W, H):
    x, y, w, h = box
    return {
        "left": x / W,
        "top": y / H,
        "right": (x + w) / W,
        "bottom": (y + h) / H
    }

@app.get("/debug/{fn}")
def debug_image(fn: str):
    return FileResponse(os.path.join(DEBUG_FOLDER, fn))

@app.post("/detect")
async def detect_card(body: dict):
    url = body.get("image_url")
    if not url:
        return {"error": "No image_url provided."}

    # fetch image
    try:
        resp = httpx.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Could not download: {e}"}

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    frame = np.array(img)
    H, W = frame.shape[:2]

    # 1) mask whites in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # white = low saturation, high value
    mask = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))

    # 2) clean it up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # 3) find contours, skip any touching the image border
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        # skip if it spans the full image (i.e. background mask leak)
        if x==0 or y==0 or x+w>=W or y+h>=H:
            continue
        if area > best_area:
            best_area, best = area, (x,y,w,h)

    if best is None:
        return {"error": "No valid white‐border region found."}

    x,y,w,h = best

    # draw debug
    dbg = frame.copy()
    cv2.rectangle(dbg, (x,y),(x+w,y+h), (0,255,0), 3)
    fn = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(DEBUG_FOLDER, fn)
    cv2.imwrite(path, cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))

    # inner = inset 10%
    inset_x, inset_y = int(w*0.1), int(h*0.1)
    inner_box = (x+inset_x, y+inset_y, w-2*inset_x, h-2*inset_y)

    return {
        "outerEdges": normalize_box((x,y,w,h), W, H),
        "innerEdges": normalize_box(inner_box,  W, H),
        "rotation": 0.0,
        "confidence": 0.99,
        "debug_image_url": f"/debug/{fn}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
