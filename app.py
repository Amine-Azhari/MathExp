from flask import Flask, render_template, request, jsonify
import base64, io
import numpy as np
from PIL import Image
import torch

from model import CNN
from connectedComps import predict_expression
from calculator import evaluate

import cv2
cv2.setNumThreads(0) #newby

app = Flask(__name__)

# Load model ONCE at startup
device = torch.device("cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("digits_ops_CNN.pth", map_location=device))
model.eval()
torch.set_num_threads(1) #newby


def decode_base64_to_gray(data_url: str) -> np.ndarray:
    header, b64data = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale
    img = img.resize((600, 168)) # newby
    return np.array(img, dtype=np.uint8)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing image"}), 400

        gray = decode_base64_to_gray(data["image"])

        expr, boxes, symbols = predict_expression(gray, model, device)

        result = evaluate(expr) if expr else ""

        return jsonify({
            "symbols": symbols,
            "expr": expr,
            "result": result
        })

    except Exception as e:
        print("ERROR in /predict:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
