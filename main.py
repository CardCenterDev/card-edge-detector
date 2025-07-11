from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import httpx
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

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(image_url, headers={"User-Agent": "EdgeDetector/1.0"})

        if response.status_code != 200:
            return {"error": f"Failed to download image. Status: {response.status_code}"}

        # Convert image to a NumPy array
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image_np = np.array(image)
        height, width = image_np.shape[:2]

        # === Placeholder logic for testing ===
        outer = {"left": 0.05, "top": 0.05, "right": 0.95, "bottom": 0.95}
        inner = {"left": 0.10, "top": 0.10, "right": 0.90, "bottom": 0.90}

        return {
            "outerEdges": outer,
            "innerEdges": inner,
            "rotation": 0.0,
            "confidence": 0.95
        }

    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

# Optional for local dev
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
