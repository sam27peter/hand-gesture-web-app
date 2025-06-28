import os
import cv2
import numpy as np
import json

# ✅ Set correct dataset directory
DATASET_DIR = "ASL_Alphabet_Dataset/asl_alphabet_train"
IMG_SIZE = 64  # ✅ Updated to 64 for better resolution
MAX_IMAGES_PER_CLASS = 200
OUTPUT_FILE = "processed_data.npz"

data = []
labels = []
label_map = {}
label_counter = 0

# ✅ Loop through folders (A-Z, del, nothing, space)
for label_name in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, label_name)
    if not os.path.isdir(class_path):
        continue

    print(f"📁 Processing label: {label_name}")
    label_map[str(label_counter)] = label_name
    image_count = 0

    for img_name in os.listdir(class_path):
        if image_count >= MAX_IMAGES_PER_CLASS:
            break

        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # ✅ Keep as RGB and resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Normalize to [0,1]

        data.append(img)
        labels.append(label_counter)
        image_count += 1

    label_counter += 1

# ✅ Convert to NumPy arrays
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # ✅ 3 channels for RGB
labels = np.array(labels)

# ✅ Save to .npz
np.savez_compressed(OUTPUT_FILE, data=data, labels=labels)
print(f"✅ All labels processed.")
print(f"✅ Saved preprocessed dataset to '{OUTPUT_FILE}' with {len(data)} samples.")

# ✅ Save label map
os.makedirs("model", exist_ok=True)
with open("model/label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=4)
print("✅ Saved label map to 'model/label_map.json'")