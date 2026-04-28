"""
Rice Disease Classification - ResNet50 Training Script
Local training with PyTorch - No Google Colab dependencies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import copy
import time

# ====== CONFIGURATION ======
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_SAVE_PATH = r"D:\rafsan\rice_resnet_model.pth"

BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10

# ====== DEVICE SETUP ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ====== TRANSFORMS ======
# ImageNet normalization
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# ====== LOAD DATA ======
print("\nLoading datasets...")
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ====== CLASS WEIGHTS FOR IMBALANCE ======
class_counts = [sum(1 for _, label in train_dataset.samples if label == i) for i in range(num_classes)]
print(f"Class distribution: {class_counts}")

weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum() * num_classes  # Normalize
weights = weights.to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# ====== MODEL SETUP ======
print("\nLoading ResNet50...")
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze early layers, fine-tune later layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer3, layer4, and fc for fine-tuning
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model = model.to(device)

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

# ====== OPTIMIZER & SCHEDULER ======
optimizer = optim.AdamW([
    {'params': model.layer3.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
    {'params': model.layer4.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
    {'params': model.fc.parameters(), 'lr': 1e-3, 'weight_decay': 0.001}
])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6
)

# ====== TRAINING LOOP ======
print("\n" + "="*60)
print("TRAINING STARTED")
print("="*60)

best_val_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # ---- Training ----
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        # Progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            progress = (batch_idx + 1) / len(train_loader) * 100
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} ({progress:.0f}%) | Loss: {loss.item():.4f}")
    
    avg_train_loss = running_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total
    
    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    epoch_time = time.time() - start_time
    
    # ---- Report ----
    print(f"\n  Epoch {epoch+1} Summary ({epoch_time:.1f}s):")
    print(f"    Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"    Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    
    # ---- Save Best Model ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        print(f"    *** New best model! Val Acc: {val_acc:.2f}% ***")
    else:
        epochs_without_improvement += 1
        print(f"    (No improvement: {epochs_without_improvement}/{PATIENCE})")
    
    # ---- Update Learning Rate ----
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"    Current LR: {current_lr:.2e}")
    
    # ---- Early Stopping ----
    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping triggered after {epoch+1} epochs!")
        break
    
    print("-" * 60)

# ====== SAVE MODEL ======
print("\n" + "="*60)
model.load_state_dict(best_model_weights)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print("="*60)
