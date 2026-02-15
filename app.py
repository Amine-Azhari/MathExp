from flask import Flask, render_template, request, jsonify
import base64
import io

import numpy as np
from PIL import Image, ImageOps

import torch

from model import CNN

app = Flask(__name__)

# Load model ONCE at startup (this is the key)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("digits_ops_CNN.pth", map_location=device))
model.eval()


def preprocess_base64_image(data_url: str) -> torch.Tensor:
    """
    data_url looks like: "data:image/png;base64,AAAA..."
    We'll decode it, convert to 28x28 grayscale, and normalize to [-1, 1]
    to match MNIST normalization ((x-0.5)/0.5).
    """
    # Split header "data:image/png;base64,"
    header, b64data = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64data)

    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale
    # Canvas: background is black, drawing is white -> MNIST is usually black bg, white digit (OK)
    # If you ever invert colors on the frontend, you may need ImageOps.invert

    # Resize to 28x28
    img = img.resize((28, 28), resample=Image.Resampling.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0  # [0,1]

    # Normalize to [-1,1] like your training: Normalize((0.5,), (0.5,))
    arr = (arr - 0.5) / 0.5

    # Shape [1, 1, 28, 28]
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing image"}), 400

    x = preprocess_base64_image(data["image"]).to(device)

    with torch.no_grad():
        out = model(x)
        pred = int(torch.argmax(out, dim=1).item())

    return jsonify({"prediction": pred})


if __name__ == "__main__":
    # debug=True auto-reloads when you edit code
    app.run(host="0.0.0.0", port=5000, debug=True)
