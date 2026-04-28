import os
import random
import shutil

print("Script is running...")

# =========================
# 1. PATHS (FIXED WITH FULL PATH)
# =========================
source_dir = r"D:\rafsan\Original Dataset"   # FULL PATH (fixes error)
train_dir = r"D:\rafsan\dataset\train"
val_dir = r"D:\rafsan\dataset\val"

split_ratio = 0.8

# =========================
# 2. LOOP THROUGH CLASSES
# =========================
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)

    # skip if not a folder
    if not os.path.isdir(category_path):
        continue

    print(f"Processing: {category}")

    # create train/val folders
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

    # get image files only
    images = [
        f for f in os.listdir(category_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if len(images) == 0:
        print(f"Warning: No images found in {category}")
        continue

    # shuffle images
    random.shuffle(images)

    # split
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # =========================
    # 3. COPY FILES
    # =========================
    for img in train_images:
        shutil.copy(
            os.path.join(category_path, img),
            os.path.join(train_dir, category, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(category_path, img),
            os.path.join(val_dir, category, img)
        )

    print(f"[OK] {category}: {len(train_images)} train, {len(val_images)} val")

print("\nDataset successfully split!")