from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os
import io
import requests
from werkzeug.utils import secure_filename
from datetime import datetime
import json
from pymongo import MongoClient
import bcrypt
import numpy as np
import joblib
from PIL import Image

# ------------------ TensorFlow Setup ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ✅ Reduce TensorFlow memory & thread usage for Render
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ------------------------------------------------------

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# ------------------ Model & Encoder Loading ------------------
MODEL_PATH = './model_artifacts/best_model.keras'
ENCODER_PATH = './model_artifacts/label_encoder.pkl'

model = None
label_encoder = None
CLASS_NAMES = None
IMG_SIZE = (128, 128)

try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}")

    if os.path.exists(ENCODER_PATH):
        print(f"Loading label encoder from: {ENCODER_PATH}")
        label_encoder = joblib.load(ENCODER_PATH)
        CLASS_NAMES = label_encoder.classes_
        print(f"✅ Label encoder loaded successfully. Classes: {CLASS_NAMES}")
    else:
        print(f"⚠️ Label encoder not found at {ENCODER_PATH}")
except Exception as e:
    print(f"❌ Error loading ML model: {e}")
# -------------------------------------------------------------

# ------------------ MongoDB Atlas Connection ------------------
try:
    client = MongoClient(
        "mongodb+srv://kummethadineshwarreddy_db_user:Din123@cluster0.mxpr4jp.mongodb.net/?appName=Cluster0"
    )
    db = client['fabric_defect_db']
    users_collection = db['users']
    print("✅ Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"❌ Error connecting to MongoDB Atlas: {e}")
    users_collection = None
# -------------------------------------------------------------

# ------------------ Helper Functions ------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_pil):
    """Prepare a PIL image for the model."""
    image_pil = image_pil.resize(IMG_SIZE)
    image_array = img_to_array(image_pil)

    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 1:
        image_array = np.concatenate([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[..., :3]

    image_batch = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input(image_batch)
    return processed_image

def predict_defect(image_pil):
    """Predict fabric defect using the trained model."""
    if model is None or label_encoder is None:
        return None, None, None

    try:
        processed_image = preprocess_image(image_pil)

        # ✅ Force CPU-only inference to prevent CUDA/memory issues
        with tf.device("/CPU:0"):
            predictions = model.predict(processed_image, verbose=0)

        prob_good = float(predictions[0][0])
        prob_defect = float(1.0 - prob_good)

        if prob_good > 0.5:
            pred_class_name = "good"
            pred_confidence = prob_good * 100.0
        else:
            pred_class_name = "defect"
            pred_confidence = prob_defect * 100.0

        all_predictions = {"good": prob_good * 100.0, "defect": prob_defect * 100.0}

        print(f"✅ Prediction: {pred_class_name} ({pred_confidence:.2f}%)")

        return pred_class_name, pred_confidence, all_predictions

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None, None
# -------------------------------------------------------------

# ------------------ Routes ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name, ext = os.path.splitext(filename)
    filename = f"{name}_{timestamp}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        image_pil = Image.open(filepath)
        pred_class, pred_confidence, all_predictions = predict_defect(image_pil)

        if pred_class is None:
            return jsonify({'success': False, 'message': 'Prediction failed'}), 500

        return jsonify({
            'success': True,
            'filename': filename,
            'class': pred_class,
            'confidence': round(pred_confidence, 2),
            'all_predictions': all_predictions
        })

    except Exception as e:
        print(f"❌ Upload prediction failed: {e}")
        return jsonify({'success': False, 'message': f'Server error: {e}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# -------------------------------------------------------------

# ------------------ Run Server ------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
# -------------------------------------------------------------
