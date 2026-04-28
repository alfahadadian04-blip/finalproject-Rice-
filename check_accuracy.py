import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# Load data
val_dataset = datasets.ImageFolder('dataset/val', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load model
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(val_dataset.classes))
model.load_state_dict(torch.load('rice_resnet_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Evaluate
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'>>> Validation Accuracy: {accuracy:.2f}%')
print(f'>>> Correct: {correct}/{total}')
