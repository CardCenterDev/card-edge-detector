from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np

app = Flask(__name__)

# --- Configuration ---
# Standard trading card dimensions (e.g., Magic: The Gathering, Pokemon)
STANDARD_CARD_WIDTH_MM = 63.5
STANDARD_CARD_HEIGHT_MM = 88.9

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_url = data.get('imageUrl')
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

        # 3. Canny edges + findContours
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # 4. Pick the largest 4-point contour
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
            # fallback: full image
            min_x, min_y, box_w, box_h = 0, 0, w, h
        else:
            min_x, min_y, box_w, box_h = cv2.boundingRect(card_contour)

        max_x, max_y = min_x + box_w, min_y + box_h

        # 5. Outer edges normalized
        outer_left   = min_x / w
        outer_top    = min_y / h
        outer_right  = max_x / w
        outer_bottom = max_y / h

        # 6. Inner edges as fixed inset
        inset_x = box_w * 0.05
        inset_y = box_h * 0.05
        inner_left   = (min_x + inset_x) / w
        inner_top    = (min_y + inset_y) / h
        inner_right  = (max_x - inset_x) / w
        inner_bottom = (max_y - inset_y) / h

        # 7. Rotation (via minAreaRect)
        rotation = 0.0
        if card_contour is not None:
            rect = cv2.minAreaRect(card_contour)
            angle = rect[2]
            # normalize to [-45,45]
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            rotation = float(f"{angle:.2f}")

        return jsonify({
            "outerEdges": {
                "left": float(f"{outer_left:.4f}"),
                "top": float(f"{outer_top:.4f}"),
                "right": float(f"{outer_right:.4f}"),
                "bottom": float(f"{outer_bottom:.4f}")
            },
            "innerEdges": {
                "left": float(f"{inner_left:.4f}"),
                "top": float(f"{inner_top:.4f}"),
                "right": float(f"{inner_right:.4f}"),
                "bottom": float(f"{inner_bottom:.4f}")
            },
            "rotation": rotation,
            "confidence": 0.9
        }), 200

    except requests.RequestException as e:
        return jsonify({"error": f"Download failed: {e}"}), 502
    except Exception as e:
        return jsonify({"error": f"Processing error: {e}"}), 500


if __name__ == '__main__':
    # for local dev only
    app.run(host='0.0.0.0', port=5000, debug=True)
