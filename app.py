from flask import Flask, request, jsonify
from card_detector import detect_card_edges_from_bytes

app = Flask(__name__)

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    img_bytes = request.files["file"].read()
    try:
        res = detect_card_edges_from_bytes(img_bytes)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render provides PORT via env var
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
