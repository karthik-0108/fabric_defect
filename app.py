from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
import joblib
from PIL import Image
from pymongo import MongoClient
import bcrypt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================================
# CONFIGURATION
# ==========================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Optimize TensorFlow for Render (CPU)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

for dev in tf.config.list_physical_devices('CPU'):
    try:
        tf.config.experimental.set_memory_growth(dev, True)
    except:
        pass

# ==========================================================
# FLASK APP SETUP
# ==========================================================
app = Flask(__name__)
app.secret_key = "render-secure-key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "./model_artifacts/best_model.keras"
ENCODER_PATH = "./model_artifacts/label_encoder.pkl"

model = None
label_encoder = None
IMG_SIZE = (128, 128)

# ==========================================================
# LOAD MODEL AND ENCODER
# ==========================================================
try:
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None

try:
    print(f"Loading label encoder from: {ENCODER_PATH}")
    label_encoder = joblib.load(ENCODER_PATH)
    print(f"✅ Label encoder loaded. Classes: {label_encoder.classes_}")
except Exception as e:
    print(f"❌ Label encoder load failed: {e}")
    label_encoder = None

# ==========================================================
# MONGODB CONNECTION
# ==========================================================
try:
    client = MongoClient(
        "mongodb+srv://kummethadineshwarreddy_db_user:Din123@cluster0.mxpr4jp.mongodb.net/?appName=Cluster0"
    )
    db = client["fabric_defect_db"]
    print("✅ Connected to MongoDB Atlas")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    db = None

# ==========================================================
# HELPERS
# ==========================================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = img_to_array(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def predict_image(image_pil):
    if model is None:
        return {"class": "N/A", "confidence": 0.0, "error": "Model not available"}

    try:
        with tf.device("/CPU:0"):
            processed = preprocess_image(image_pil)
            preds = model.predict(processed, verbose=0)[0]

        prob_good = float(preds[0])
        prob_defect = 1.0 - prob_good
        label = "good" if prob_good >= 0.5 else "defect"
        conf = max(prob_good, prob_defect) * 100.0

        return {"class": label, "confidence": round(conf, 2), "probs": {"good": prob_good, "defect": prob_defect}}
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return {"class": "N/A", "confidence": 0.0, "error": str(e)}

# ==========================================================
# AUTHENTICATION ROUTES
# ==========================================================
@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"success": False, "message": "Username and password required"}), 400

        if db["users"].find_one({"username": username}):
            return jsonify({"success": False, "message": "Username already exists"}), 400

        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        db["users"].insert_one({
            "username": username,
            "password": hashed,
            "created_at": datetime.now().isoformat()
        })
        return jsonify({"success": True, "message": "Signup successful"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        user = db["users"].find_one({"username": username})
        if user and bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
            session["username"] = username
            return jsonify({"success": True, "message": "Login successful"})
        else:
            return jsonify({"success": False, "message": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/logout")
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out"})

# ==========================================================
# FRONTEND + ML ROUTES
# ==========================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "message": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "message": "Empty filename"}), 400
        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{os.path.splitext(filename)[0]}_{timestamp}.png"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        with Image.open(filepath) as img:
            result = predict_image(img)

        return jsonify({"success": True, "filename": filename, "prediction": result})
    except MemoryError:
        return jsonify({"success": False, "message": "Server ran out of memory. Try a smaller image."}), 507
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/uploads/<filename>")
def serve_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
