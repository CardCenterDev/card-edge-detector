# main.py
from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np

app = Flask(__name__)

# --- Configuration ---
STANDARD_CARD_WIDTH_MM = 63.5
STANDARD_CARD_HEIGHT_MM = 88.9

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json(force=True)  # always parse JSON
    # accept either key
    image_url = data.get('imageUrl') or data.get('image_url')
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        # 1. Download the image
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        buf = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from URL.")

        h, w, _ = img.shape

        # 2. Grayscale + blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Canny edges + contours
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # 4. Largest 4-point rectangle
        card_contour = None
        max_area = 0
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area and area > (w * h * 0.1):
                    card_contour = approx
                    max_area = area

        if card_contour is None:
            # fallback → full image
            min_x, min_y, box_w, box_h = 0, 0, w, h
        else:
            min_x, min_y, box_w, box_h = cv2.boundingRect(card_contour)

        max_x, max_y = min_x + box_w, min_y + box_h

        # 5. Normalize outer edges
        outer = {
            "left":   round(min_x / w, 4),
            "top":    round(min_y / h, 4),
            "right":  round(max_x / w, 4),
            "bottom": round(max_y / h, 4),
        }

        # 6. Inner edges (5% inset)
        inset_x = box_w * 0.05
        inset_y = box_h * 0.05
        inner = {
            "left":   round((min_x + inset_x) / w, 4),
            "top":    round((min_y + inset_y) / h, 4),
            "right":  round((max_x - inset_x) / w, 4),
            "bottom": round((max_y - inset_y) / h, 4),
        }

        # 7. Rotation
        rotation = 0.0
        if card_contour is not None:
            rect = cv2.minAreaRect(card_contour)
            angle = rect[2]
            # normalize to [-45,45]
            if angle < -45: angle += 90
            elif angle > 45: angle -= 90
            rotation = round(angle, 2)

        return jsonify({
            "outerEdges": outer,
            "innerEdges": inner,
            "rotation": rotation,
            "confidence": 0.9
        }), 200

    except requests.RequestException as e:
        return jsonify({"error": f"Download failed: {e}"}), 502
    except Exception as e:
        return jsonify({"error": f"Processing error: {e}"}), 500


if __name__ == '__main__':
    # for local dev:
    app.run(host='0.0.0.0', port=5000, debug=True)
