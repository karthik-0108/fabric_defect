from flask import Flask, render_template, request, session, jsonify, send_from_directory
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

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# --- ML Model and Encoder Paths ---
MODEL_PATH = './model_artifacts/best_model.keras'
ENCODER_PATH = './model_artifacts/label_encoder.pkl'

model = None
label_encoder = None
CLASS_NAMES = None
IMG_SIZE = (128, 128)

# Load ML model and encoder
try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

    if os.path.exists(ENCODER_PATH):
        print(f"Loading label encoder from: {ENCODER_PATH}")
        label_encoder = joblib.load(ENCODER_PATH)
        CLASS_NAMES = label_encoder.classes_
        print(f"Label encoder loaded successfully. Classes: {CLASS_NAMES}")
    else:
        print(f"Warning: Label encoder not found at {ENCODER_PATH}")
except Exception as e:
    print(f"Error loading ML model: {e}")

# --- MongoDB Atlas Connection ---
try:
    client = MongoClient("mongodb+srv://kummethadineshwarreddy_db_user:Din123@cluster0.mxpr4jp.mongodb.net/?appName=Cluster0")
    db = client['fabric_defect_db']
    users_collection = db['users']
    print("Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB Atlas: {e}")
    users_collection = None

# --- History Management ---
HISTORY_FILE = 'history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:50]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_pil):
    image_pil = image_pil.resize(IMG_SIZE)
    image_array = img_to_array(image_pil)

    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 1:
        image_array = np.concatenate([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[..., :3]

    image_batch = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_batch)

def predict_defect(image_pil):
    if model is None:
        return None, None, None

    try:
        processed_image = preprocess_image(image_pil)
        predictions = model.predict(processed_image, verbose=0)
        prob_good = float(predictions[0][0])
        prob_defect = 1.0 - prob_good

        if prob_good > 0.5:
            pred_class_name = "good"
            pred_confidence = prob_good * 100.0
        else:
            pred_class_name = "defect"
            pred_confidence = prob_defect * 100.0

        all_predictions = {
            "good": round(prob_good * 100.0, 2),
            "defect": round(prob_defect * 100.0, 2)
        }
        return pred_class_name, pred_confidence, all_predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/check-auth')
def check_auth():
    if 'username' in session:
        return jsonify({'authenticated': True, 'username': session['username']})
    return jsonify({'authenticated': False})

@app.route('/signup', methods=['POST'])
def signup():
    try:
        if users_collection is None:
            return jsonify({'success': False, 'message': 'Database connection error'}), 500

        name = request.form.get('name', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not all([name, username, password]):
            return jsonify({'success': False, 'message': 'All fields are required'}), 400

        if users_collection.find_one({'username': username}):
            return jsonify({'success': False, 'message': 'Username already exists'}), 400

        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_doc = {
            'name': name,
            'username': username,
            'password': password_hash.decode('utf-8'),
            'created_at': datetime.now().isoformat()
        }

        result = users_collection.insert_one(user_doc)
        session['username'] = username
        session['user_id'] = str(result.inserted_id)
        return jsonify({'success': True, 'username': username, 'message': 'Account created successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = users_collection.find_one({'username': username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            session['username'] = username
            session['user_id'] = str(user['_id'])
            return jsonify({'success': True, 'username': username})
        return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401

    upload_type = request.form.get('upload_type')
    prediction_result = None
    filename = None

    try:
        if upload_type == 'file':
            file = request.files.get('file')
            if not file or not allowed_file(file.filename):
                return jsonify({'success': False, 'message': 'Invalid file'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_bytes = file.read()
            file.seek(0)
            file.save(filepath)
            image_pil = Image.open(io.BytesIO(image_bytes))
            pred_class, pred_confidence, all_predictions = predict_defect(image_pil)

            prediction_result = {
                'class': pred_class or 'N/A',
                'confidence': round(pred_confidence or 0, 2),
                'is_defect': (pred_class or '').lower() != 'good',
                'all_predictions': all_predictions or {}
            }

        response_data = {
            'success': True,
            'message': 'File processed successfully',
            'filename': filename,
            'prediction': prediction_result
        }

        save_history({
            'username': session['username'],
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_result
        })

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Run Application ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
