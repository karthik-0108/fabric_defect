# Fabric Defect Detection

A web application for uploading and managing fabric images for defect detection analysis. Built with Flask, HTML, and CSS.

## Features

- **User Authentication**: Secure login system
- **Image Upload**: Upload images from:
  - Local file system (browse files)
  - Online URLs (download images from web)
- **History Tracking**: View all previously uploaded images
- **Modern UI**: Beautiful, responsive design with gradient backgrounds

## Installation

1. Clone or download this project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Default Login Credentials

- **Username**: `admin` | **Password**: `password123`
- **Username**: `user` | **Password**: `user123`

## Project Structure

```
fabric-defect-detection/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── templates/            # HTML templates
│   └── index.html        # Single page application (SPA) - combines all pages
│
├── static/               # Static files
│   └── style.css         # Main stylesheet
│
├── uploads/             # Directory for uploaded images (created automatically)
└── history.json         # JSON file storing upload history (created automatically)
```

## Usage

1. **Login**: Use the default credentials to log in
2. **Upload Images**: 
   - Go to the Home page
   - Choose between "Browse File" or "Upload from URL"
   - Select your image or paste an image URL
   - Click upload
3. **View History**: Check the History page to see all your previous uploads

## Notes

- Supported image formats: PNG, JPG, JPEG, GIF, BMP, WEBP
- Maximum file size: 16MB
- Uploaded images are stored in the `uploads/` directory
- History is stored in `history.json` (keeps last 50 entries)

## Security Notes

⚠️ **Important**: This is a demo application. For production use:
- Change the `secret_key` in `app.py`
- Use a proper database instead of JSON files
- Implement proper password hashing
- Add CSRF protection
- Use environment variables for sensitive data

