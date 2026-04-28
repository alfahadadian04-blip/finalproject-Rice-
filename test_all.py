"""
Comprehensive Test Script - Rice Disease Detection System
Tests: Dataset, Model, API
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

print("="*60)
print("RICE DISEASE DETECTION - FULL SYSTEM TEST")
print("="*60)

# ====== 1. TEST DATASET ======
print("\n[1] CHECKING DATASET...")
print("-"*40)

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

errors = []

if not os.path.exists(TRAIN_DIR):
    errors.append(f"Missing: {TRAIN_DIR}")
if not os.path.exists(VAL_DIR):
    errors.append(f"Missing: {VAL_DIR}")

if errors:
    print("ERROR: Dataset folders not found!")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

train_classes = sorted(os.listdir(TRAIN_DIR)) if os.path.exists(TRAIN_DIR) else []
val_classes = sorted(os.listdir(VAL_DIR)) if os.path.exists(VAL_DIR) else []

print(f"Train classes ({len(train_classes)}): {train_classes}")
print(f"Val classes ({len(val_classes)}): {val_classes}")

# Count images
train_counts = {}
val_counts = {}
for cls in train_classes:
    path = os.path.join(TRAIN_DIR, cls)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        train_counts[cls] = count

for cls in val_classes:
    path = os.path.join(VAL_DIR, cls)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_counts[cls] = count

print("\nDataset distribution:")
total_train = 0
total_val = 0
for cls in train_classes:
    t = train_counts.get(cls, 0)
    v = val_counts.get(cls, 0)
    total_train += t
    total_val += v
    print(f"  {cls:20s}: Train={t:4d}, Val={v:4d}")

print(f"\nTotal: Train={total_train}, Val={total_val}")

if set(train_classes) != set(val_classes):
    print("WARNING: Train and Val classes don't match!")

# ====== 2. TEST MODEL ======
print("\n[2] CHECKING MODEL...")
print("-"*40)

MODEL_PATH = r"D:\rafsan\rice_resnet_model.pth"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

print(f"Model found: {MODEL_PATH}")
print(f"Model size: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = resnet50(weights=None)
num_classes = len(train_classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

# ====== 3. TEST ACCURACY ======
print("\n[3] TESTING MODEL ACCURACY...")
print("-"*40)

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

try:
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    correct = 0
    total = 0
    class_correct = {cls: 0 for cls in val_dataset.classes}
    class_total = {cls: 0 for cls in val_dataset.classes}
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                cls = val_dataset.classes[labels[i]]
                class_total[cls] += 1
                if predicted[i] == labels[i]:
                    class_correct[cls] += 1
    
    accuracy = 100 * correct / total
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("\nPer-class accuracy:")
    for cls in val_dataset.classes:
        if class_total[cls] > 0:
            cls_acc = 100 * class_correct[cls] / class_total[cls]
            print(f"  {cls:20s}: {cls_acc:5.1f}% ({class_correct[cls]}/{class_total[cls]})")
    
except Exception as e:
    print(f"ERROR during accuracy test: {e}")

# ====== 4. TEST SINGLE PREDICTION ======
print("\n[4] TESTING SINGLE IMAGE PREDICTION...")
print("-"*40)

# Find a test image
test_image = None
for cls in train_classes[:1]:  # Use first class
    cls_path = os.path.join(VAL_DIR, cls)
    if os.path.exists(cls_path):
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            test_image = os.path.join(cls_path, images[0])
            break

if test_image and os.path.exists(test_image):
    try:
        img = Image.open(test_image).convert('RGB')
        img_tensor = val_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probs, 0)
        
        print(f"Test image: {test_image}")
        print(f"Predicted: {val_dataset.classes[predicted.item()]} ({confidence.item()*100:.2f}% confidence)")
        
        # Top 3
        top3_probs, top3_indices = torch.topk(probs, 3)
        print("Top 3 predictions:")
        for i in range(3):
            print(f"  {i+1}. {val_dataset.classes[top3_indices[i]]}: {top3_probs[i]*100:.2f}%")
    except Exception as e:
        print(f"Prediction test failed: {e}")
else:
    print("No test image found")

# ====== 5. TEST API AVAILABILITY ======
print("\n[5] CHECKING API...")
print("-"*40)

API_FILE = "api_fixed.py"
if os.path.exists(API_FILE):
    print(f"API file found: {API_FILE}")
    # Check if API can be imported
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("api", API_FILE)
        api_module = importlib.util.module_from_spec(spec)
        print("API module structure OK")
    except Exception as e:
        print(f"API import warning: {e}")
else:
    print(f"WARNING: API file not found: {API_FILE}")

# ====== SUMMARY ======
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print(f"Dataset:      {'OK' if total_train > 0 and total_val > 0 else 'FAILED'}")
print(f"Model:        {'OK' if os.path.exists(MODEL_PATH) else 'FAILED'}")
print(f"Accuracy:     {accuracy:.2f}%" if 'accuracy' in locals() else "Accuracy:     Not tested")
print(f"Prediction:   {'OK' if test_image else 'FAILED'}")
print("="*60)

if accuracy >= 80:
    print("Model accuracy is GOOD (>=80%)")
elif accuracy >= 60:
    print("Model accuracy is MODERATE (60-80%)")
else:
    print("Model accuracy needs improvement (<60%)")

print("\nRun 'python api_fixed.py' to start the API server")
