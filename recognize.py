import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# ====== LOAD MODEL ======
model_path = "rice_resnet_model.pth"

# class names (must match training - 6 classes)
classes = [
    "Healthy",
    "Leaf Blight",
    "Rice Blast",
    "Rice Leaffolder",
    "Rice Stripes",
    "Rice Tungro"
]

# load model
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# ====== IMAGE TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ====== GET IMAGE PATH FROM TERMINAL ======
if len(sys.argv) < 2:
    print("❌ Please provide image path")
    print('Example: python recognize.py "D:\\image.jpg"')
    exit()

image_path = sys.argv[1]

# check if file exists
if not os.path.exists(image_path):
    print("❌ File not found:", image_path)
    exit()

# ====== PREDICT ======
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

result = classes[predicted.item()]

print("✅ Prediction:", result)