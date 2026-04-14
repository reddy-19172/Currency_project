from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

IMG_SIZE = 128

# ✅ Load model once
model = load_model("model.h5")
print("🔥 Warming up model...")

# ✅ Warmup (prevents delay on first request)
dummy = np.zeros((1, 128, 128, 1))
model.predict(dummy)

print("✅ Model ready")

# ✅ Preprocess
def preprocess_image(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = gray / 255.0
    gray = gray.reshape(IMG_SIZE, IMG_SIZE, 1)

    return gray

# ✅ Home route
@app.route("/")
def home():
    return "✅ Banknote API Running"

# ✅ Predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("📩 Request received")

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        img = preprocess_image(file)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]
        print("🔢 Prediction:", prediction)

        result = "Real Banknote ✅" if prediction > 0.6 else "Fake Banknote ❌"

        return jsonify({"result": result})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)})

# ✅ Render port fix
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)