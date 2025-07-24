# card_detector.py (Roboflow-powered)

from inference_sdk import InferenceHTTPClient
from PIL import Image
import io

# Load from environment or hardcode here if testing locally
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "your_api_key_here")
MODEL_ID = "cc-rhvdm/1"

# Initialize Roboflow API client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

def detect_card_edges_from_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))

    # Run inference via Roboflow
    result = client.infer(image, model_id=MODEL_ID)

    # Convert predicted polygons to bounding boxes
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

    # Return in your app’s expected format
    return edges
