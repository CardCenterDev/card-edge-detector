# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from card_detector import detect_card_edges_from_bytes

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("DETECTOR_API_KEY")

@app.before_request
def require_api_key():
    # Check for key in header "x-api-key"
    incoming = request.headers.get("x-api-key")
    if incoming != API_KEY:
        return jsonify({"error":"unauthorized"}), 401

@app.route("/detect", methods=["POST"])
def detect():
    # now, guaranteed to have right key
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    img_bytes = request.files["file"].read()
    try:
        res = detect_card_edges_from_bytes(img_bytes)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
