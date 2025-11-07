from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
import os
import io
import requests
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import base64
from pymongo import MongoClient
from bson import ObjectId
import bcrypt
import numpy as np
import joblib
from PIL import Image

# Import TensorFlow and Keras utilities
# Suppress TensorFlow startup messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)


# --- Load ML Model and Label Encoder ---
# Define the paths to your saved model and encoder from the corrected notebook
MODEL_PATH = './model_artifacts/best_model (1).keras'
ENCODER_PATH = './model_artifacts/label_encoder (3).pkl'

# Initialize model and encoder variables
model = None
label_encoder = None
CLASS_NAMES = None
IMG_SIZE = (128, 128)

# Try to load the model and encoder
try:
    if os.path.exists(MODEL_PATH):
        MODEL_FILE = MODEL_PATH
        print(f"Loading model from: {MODEL_FILE}")
        model = load_model(MODEL_FILE)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Could not find model at '{MODEL_PATH}'")
        print("This file should be in a 'model_artifacts' folder in the same directory as app.py")
        print("Application will run without ML predictions.")
    
    if os.path.exists(ENCODER_PATH):
        print(f"Loading label encoder from: {ENCODER_PATH}")
        label_encoder = joblib.load(ENCODER_PATH)
        CLASS_NAMES = label_encoder.classes_
        print(f"Label encoder loaded successfully. Classes: {CLASS_NAMES}")
    else:
        print(f"Warning: Could not find label encoder at '{ENCODER_PATH}'")
        print("This file should be in a 'model_artifacts' folder.")
        print("Application will run without ML predictions.")
except Exception as e:
    print(f"Error loading ML model: {e}")
    print("Application will run without ML predictions.")

def preprocess_image(image_pil):
    """
    Prepares a PIL image for the MobileNetV2 model.
    1. Resizes to (128, 128)
    2. Converts to a NumPy array
    3. Creates a "batch" of 1
    4. Applies MobileNetV2 preprocessing
    """
    # Resize the image
    image_pil = image_pil.resize(IMG_SIZE)
    
    # Convert PIL image to NumPy array
    image_array = img_to_array(image_pil)
    
    # Ensure it's 3 channels (in case of grayscale uploads)
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 1:
        image_array = np.concatenate([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[..., :3]  # Drop alpha channel
    
    # Expand dimensions to create a batch (1, 128, 128, 3)
    image_batch = np.expand_dims(image_array, axis=0)
    
    # Apply the same preprocessing as in training
    processed_image = preprocess_input(image_batch)
    
    return processed_image

def predict_defect(image_pil):
    """
    Predicts fabric defects from a PIL image for a BINARY model.
    Returns: (class_name, confidence, all_predictions)
    """
    if model is None or label_encoder is None:
        return None, None, None
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_pil)
        
        # Get model's prediction
        # For a binary model, this will be a single value, e.g., [[0.95]]
        predictions = model.predict(processed_image, verbose=0)
        
        # Get the single probability value for class 1 ('good')
        prob_good = float(predictions[0][0])
        prob_defect = float(1.0 - prob_good)
        
        # Determine the predicted class
        if prob_good > 0.5:
            pred_class_name = "good"
            pred_confidence = prob_good * 100.0
        else:
            pred_class_name = "defect"
            pred_confidence = prob_defect * 100.0
            
        # Create the dictionary for all predictions
        all_predictions = {
            "good": prob_good * 100.0,
            "defect": prob_defect * 100.0
        }
        
        return pred_class_name, pred_confidence, all_predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

# MongoDB connection
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['fabric_defect_db']
    users_collection = db['users']
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    users_collection = None

# History storage (in production, use a database)
HISTORY_FILE = 'history.json'

def load_history():
    """Load history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_history(entry):
    """Save history entry to JSON file"""
    history = load_history()
    history.insert(0, entry)  # Add to beginning
    # Keep only last 50 entries
    history = history[:50]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_image_from_url(url):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            return None, "URL does not point to an image"
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        extension = url.rsplit('.', 1)[1].lower() if '.' in url else 'jpg'
        if extension not in ALLOWED_EXTENSIONS:
            extension = 'jpg'
        filename = f"url_{timestamp}.{extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save image
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return filename, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Main page - single page application"""
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    """Signup API endpoint"""
    try:
        if users_collection is None:
            return jsonify({'success': False, 'message': 'MongoDB connection error. Please make sure MongoDB is running on localhost:27017'}), 500
        
        name = request.form.get('name', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # Validation
        if not name or not username or not password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        
        if len(username) < 3:
            return jsonify({'success': False, 'message': 'Username must be at least 3 characters'}), 400
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
        
        # Check if username already exists
        if users_collection.find_one({'username': username}):
            return jsonify({'success': False, 'message': 'Username already exists'}), 400
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create user document
        user_doc = {
            'name': name,
            'username': username,
            'password': password_hash.decode('utf-8'),
            'created_at': datetime.now().isoformat()
        }
        
        result = users_collection.insert_one(user_doc)
        if result.inserted_id:
            session['username'] = username
            session['user_id'] = str(result.inserted_id)
            return jsonify({
                'success': True, 
                'username': username,
                'message': 'Account created successfully!'
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to create account'}), 500
    except Exception as e:
        app.logger.error(f'Signup error: {str(e)}')
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/login', methods=['POST'])
def login():
    """Login API endpoint"""
    try:
        if users_collection is None:
            return jsonify({'success': False, 'message': 'MongoDB connection error. Please make sure MongoDB is running on localhost:27017'}), 500
        
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password are required'}), 400
        
        # Find user in MongoDB
        user = users_collection.find_one({'username': username})
        
        if user:
            # Verify password
            stored_password = user['password'].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                session['username'] = username
                session['user_id'] = str(user['_id'])
                return jsonify({'success': True, 'username': username})
            else:
                return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
    except Exception as e:
        app.logger.error(f'Login error: {str(e)}')
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.pop('username', None)
    return jsonify({'success': True})

@app.route('/api/check-auth')
def check_auth():
    """Check if user is authenticated"""
    if 'username' in session:
        return jsonify({'authenticated': True, 'username': session['username']})
    return jsonify({'authenticated': False})

@app.route('/api/history')
def get_history():
    """Get history data as JSON"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    history_data = load_history()
    # Filter history for current user
    user_history = [entry for entry in history_data if entry.get('username') == session['username']]
    
    return jsonify(user_history)

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload from file or URL and perform defect detection"""
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    upload_type = request.form.get('upload_type')
    
    if upload_type == 'file':
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Read image for prediction
            image_bytes = file.read()
            file.seek(0)  # Reset file pointer
            file.save(filepath)
            
            # Perform defect detection
            prediction_result = None
            try:
                image_pil = Image.open(io.BytesIO(image_bytes))
                pred_class, pred_confidence, all_predictions = predict_defect(image_pil)
                
                if pred_class is not None:
                    prediction_result = {
                        'class': pred_class,
                        'confidence': round(pred_confidence, 2),
                        'is_defect': pred_class.lower() != 'good',
                        'all_predictions': all_predictions
                    }
            except Exception as e:
                print(f"Prediction error: {e}")
                prediction_result = None
            
            # Save to history
            entry = {
                'id': len(load_history()) + 1,
                'username': session['username'],
                'filename': filename,
                'upload_type': 'file',
                'timestamp': datetime.now().isoformat(),
                'status': 'uploaded',
                'prediction': prediction_result
            }
            save_history(entry)
            
            response_data = {
                'success': True,
                'message': 'File uploaded successfully',
                'filename': filename
            }
            
            if prediction_result:
                response_data['prediction'] = prediction_result
            
            return jsonify(response_data)
        else:
            return jsonify({'success': False, 'message': 'Invalid file type'}), 400
    
    elif upload_type == 'url':
        # Handle URL upload
        image_url = request.form.get('image_url', '').strip()
        if not image_url:
            return jsonify({'success': False, 'message': 'No URL provided'}), 400
        
        filename, error = download_image_from_url(image_url)
        if error:
            return jsonify({'success': False, 'message': error}), 400
        
        # Perform defect detection on downloaded image
        prediction_result = None
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'rb') as f:
                image_bytes = f.read()
            image_pil = Image.open(io.BytesIO(image_bytes))
            pred_class, pred_confidence, all_predictions = predict_defect(image_pil)
            
            if pred_class is not None:
                prediction_result = {
                    'class': pred_class,
                    'confidence': round(pred_confidence, 2),
                    'is_defect': pred_class.lower() != 'good',
                    'all_predictions': all_predictions
                }
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction_result = None
        
        # Save to history
        entry = {
            'id': len(load_history()) + 1,
            'username': session['username'],
            'filename': filename,
            'upload_type': 'url',
            'source_url': image_url,
            'timestamp': datetime.now().isoformat(),
            'status': 'uploaded',
            'prediction': prediction_result
        }
        save_history(entry)
        
        response_data = {
            'success': True,
            'message': 'Image downloaded successfully',
            'filename': filename
        }
        
        if prediction_result:
            response_data['prediction'] = prediction_result
        
        return jsonify(response_data)
    
    return jsonify({'success': False, 'message': 'Invalid upload type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

