import os
import io
from PIL import Image
from inference_sdk import InferenceHTTPClient

# Load Roboflow credentials
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "your_api_key_here")
MODEL_ID = "cc-rhvdm/1"

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

def resize_image_if_large(image_bytes, max_width=1280):
    """
    Resizes the image down if it's wider than max_width.
    Reduces memory usage before sending to Roboflow.
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.width > max_width:
        aspect_ratio = image.height / image.width
        new_height = int(max_width * aspect_ratio)
        image = image.resize((max_width, new_height))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()
    return image_bytes

def detect_card_edges_from_bytes(image_bytes):
    """
    Accepts raw image bytes, sends to Roboflow, parses result.
    Returns bounding box data for 'outer_edge' and 'inner_border'.
    """
    image_bytes = resize_image_if_large(image_bytes)  # Optional but helps on Render Free

    # Use PIL.Image for inference
    image = Image.open(io.BytesIO(image_bytes))

    # Run inference via Roboflow
    result = client.infer(image, model_id=MODEL_ID)

    # Convert polygon predictions into bounding boxes
    edges = {}
    for prediction in result.get("predictions", []):
        label = prediction.get("class")
        if label in ("outer_edge", "inner_border"):
            points = prediction.get("points", [])
            if not points:
                continue
            xs = [pt["x"] for pt in points]
            ys = [pt["y"] for pt in points]
            edges[label] = {
                "top": int(round(min(ys))),
                "bottom": int(round(max(ys))),
                "left": int(round(min(xs))),
                "right": int(round(max(xs)))
            }

    return edges
