"""
Rice Disease Detection API - Fixed Version
Port: 5005
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import io
import os
import sys

# Create Flask app
app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = r"D:\rafsan\rice_resnet_model.pth"
PORT = 5005

# Class labels
CLASSES = [
    'Healthy',
    'Leaf Blight', 
    'Rice Blast',
    'Rice Leaffolder',
    'Rice Stripes',
    'Rice Tungro'
]

# Device setup
device = torch.device("cpu")
print(f"[INFO] Using device: {device}")

# Load model
model = None

def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model not found at {MODEL_PATH}")
            return False
        
        print("[INFO] Loading ResNet50 model...")
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"[SUCCESS] Model loaded with {len(CLASSES)} classes")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': CLASSES,
        'device': str(device)
    })

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        file = request.files['image']
        
        # Load image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probs, 0)
        
        # Get top 3
        top3_probs, top3_indices = torch.topk(probs, 3)
        top3 = [
            {'class': CLASSES[idx], 'confidence': round(float(prob) * 100, 2)}
            for idx, prob in zip(top3_indices, top3_probs)
        ]
        
        result = {
            'prediction': CLASSES[predicted.item()],
            'confidence': round(float(confidence.item()) * 100, 2),
            'top3': top3,
            'status': 'success'
        }
        
        print(f"[PREDICT] {result['prediction']} ({result['confidence']}%)")
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

# Main
if __name__ == '__main__':
    print("="*50)
    print("Rice Disease Detection API")
    print("="*50)
    
    # Load model
    if not load_model():
        print("[WARNING] Starting without model - predictions will fail")
    
    print("="*60)
    print(f"Server running at http://127.0.0.1:{PORT}")
    print("="*60)
    print(f"Use this URL to test: http://127.0.0.1:{PORT}/health")
    print("="*60)
    print("Press Ctrl+C to stop\n")
    
    # Run server
    try:
        app.run(
            host='127.0.0.1',
            port=PORT,
            debug=False,
            threaded=False,
            use_reloader=False
        )
    except Exception as e:
        print(f"\n[ERROR] Server error: {e}")
        sys.exit(1)
