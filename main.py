from fastapi import FastAPI, Query
from pydantic import BaseModel
import requests
import numpy as np
import cv2
from typing import Optional
from fastapi.responses import JSONResponse

app = FastAPI()

class EdgeResult(BaseModel):
    outerEdges: dict
    innerEdges: dict
    rotation: Optional[float] = 0.0
    confidence: Optional[float] = 0.95

def fetch_image_from_url(url):
    resp = requests.get(url, stream=True).raw
    img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def detect_card_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(card_contour)

    img_h, img_w = image.shape[:2]
    outerEdges = {
        "left": round(x / img_w, 3),
        "top": round(y / img_h, 3),
        "right": round((x + w) / img_w, 3),
        "bottom": round((y + h) / img_h, 3)
    }

    # Inner edge logic placeholder — approximate for now
    innerEdges = {
        "left": round((x + 0.05 * w) / img_w, 3),
        "top": round((y + 0.05 * h) / img_h, 3),
        "right": round((x + 0.95 * w) / img_w, 3),
        "bottom": round((y + 0.95 * h) / img_h, 3)
    }

    return {
        "outerEdges": outerEdges,
        "innerEdges": innerEdges,
        "rotation": 0.0,
        "confidence": 0.95
    }

@app.get("/analyze", response_model=EdgeResult)
async def analyze_card(image_url: str = Query(...)):
    try:
        img = fetch_image_from_url(image_url)
        result = detect_card_edges(img)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
