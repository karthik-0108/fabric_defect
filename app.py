from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os
import io
import requests
from werkzeug.utils import secure_filename
from datetime import datetime
from pymongo import MongoClient
import bcrypt
import numpy as np
import joblib
from PIL import Image

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# === App Config ===
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Model and Encoder Paths ===
MODEL_PATH = './model_artifacts/best_model.keras'
ENCODER_PATH = './model_artifacts/label_encoder.pkl'
IMG_SIZE = (128, 128)

# === Load Model ===
model = None
label_encoder = None
CLASS_NAMES = None
try:
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
    label_encoder = joblib.load(ENCODER_PATH)
    CLASS_NAMES = label_encoder.classes_
    print(f"✅ Label encoder loaded. Classes: {CLASS_NAMES}")
except Exception as e:
    print(f"⚠️ Error loading ML model: {e}")

# === MongoDB Atlas Connection ===
try:
    client = MongoClient("mongodb+srv://kummethadineshwarreddy_db_user:Din123@cluster0.mxpr4jp.mongodb.net/?appName=Cluster0")
    db = client['fabric_defect_db']
    users_collection = db['users']
    print("✅ Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"⚠️ Error connecting to MongoDB Atlas: {e}")
    users_collection = None


# === Helper Functions ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_pil):
    image_pil = image_pil.resize(IMG_SIZE)
    image_array = img_to_array(image_pil)
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[..., :3]
    image_batch = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_batch)

def predict_defect(image_pil):
    if model is None or label_encoder is None:
        return None, None, None
    try:
        processed_image = preprocess_image(image_pil)
        predictions = model.predict(processed_image, verbose=0)
        prob_good = float(predictions[0][0])
        prob_defect = float(1.0 - prob_good)
        if prob_good > 0.5:
            pred_class = "good"
            confidence = prob_good * 100
        else:
            pred_class = "defect"
            confidence = prob_defect * 100
        all_predictions = {"good": prob_good * 100, "defect": prob_defect * 100}
        return pred_class, confidence, all_predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None


# === Frontend Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    if users_collection is None:
        return jsonify({'success': False, 'message': 'Database connection error'}), 500

    name = request.form.get('name', '').strip()
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()

    if not all([name, username, password]):
        return jsonify({'success': False, 'message': 'All fields required'}), 400

    if users_collection.find_one({'username': username}):
        return jsonify({'success': False, 'message': 'Username already exists'}), 400

    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({
        'name': name,
        'username': username,
        'password': password_hash.decode('utf-8'),
        'created_at': datetime.now().isoformat()
    })
    session['username'] = username
    return jsonify({'success': True, 'username': username})

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    user = users_collection.find_one({'username': username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        session['username'] = username
        return jsonify({'success': True, 'username': username})
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/logout')
def logout():
    session.pop('username', None)
    return jsonify({'success': True})


# === File Upload Route ===
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401

    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image_pil = Image.open(filepath)
    pred_class, confidence, all_predictions = predict_defect(image_pil)

    return jsonify({
        'success': True,
        'filename': filename,
        'prediction': {
            'class': pred_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        }
    })


# === Serve Uploaded Files ===
@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# === Render Port Configuration ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
