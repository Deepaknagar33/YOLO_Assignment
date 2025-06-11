#!/usr/bin/env python
# coding: utf-8

# # YOLOv8 Dataset Preparation & Augmentation
# 
# ## 📁 1. Directory Setup
# - Define input and output paths.
# - Create subdirectories:
#   - `images/` for image files.
#   - `labels/` for YOLO-format annotation files.
# - Split data into `train`, `valid`, and `test`.
# 
# ## 🔁 2. Data Augmentation using Albumentations
# - Augmentation techniques used:
#   - `HorizontalFlip(p=0.5)`
#   - `RandomBrightnessContrast(p=0.3)`
#   - `Rotate(limit=15, p=0.2)`
#   - `MotionBlur(p=0.1)`
# - Bounding boxes are preserved using `bbox_params`.
# 
# ## 📦 3. VOC to YOLO Format Conversion
# - Each bounding box from Pascal VOC is converted to YOLO format:
#   - `x_center = (xmin + xmax)/2`
#   - `y_center = (ymin + ymax)/2`
#   - `width = xmax - xmin`
#   - `height = ymax - ymin`
# - Normalized by image width and height.
# 
# ## 📄 4. XML Annotation Parsing
# - Parses `.xml` files for bounding boxes and labels.
# - Filters only two classes:
#   - `space-empty` → class 0
#   - `space-occupied` → class 1
# 
# ## ✍️ 5. Save YOLO Labels
# - Writes `.txt` file per image in the YOLO format:
# 
# ## 🔁 6. Process Images
# For each image in `train`, `valid`, and `test`:
# 1. Parse and convert the annotation.
# 2. Save original image and its label.
# 3. Generate **2 augmented versions** using the defined pipeline.
# 4. Save augmented images and labels.
# 
# ## ✅ 7. Final Result
# Each original image has:
# - 1 original + 2 augmented images.
# - Corresponding `.txt` YOLO label files saved in appropriate split folders.
# 
# 

# In[3]:


import os
import glob
import cv2
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import albumentations as A

# Paths
INPUT_ROOT = "/home/user/YOLO_Assignment/Task_1/Dataset/pklot_raw"         
OUTPUT_ROOT = "/home/user/YOLO_Assignment/Task_1/Dataset/dataset_yolo"     
SPLITS = ['train', 'valid', 'test']

# Create output folders
for split in SPLITS:
    os.makedirs(f"{OUTPUT_ROOT}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/{split}/labels", exist_ok=True)

# Augmentation pipeline
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.2),
    A.MotionBlur(p=0.1)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# VOC to YOLO box conversion
def convert_bbox_voc_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    return [x_center * dw, y_center * dh, width * dw, height * dh]

# Parse XML annotation
def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    boxes, labels = [], []
    for obj in root.findall('object'):
        label = obj.find('name').text.strip().lower()
        if label not in ['space-empty', 'space-occupied']:
            continue
        class_id = 0 if label == 'space-empty' else 1
        xml_box = obj.find('bndbox')
        box = [int(xml_box.find('xmin').text), int(xml_box.find('ymin').text),
               int(xml_box.find('xmax').text), int(xml_box.find('ymax').text)]
        yolo_box = convert_bbox_voc_to_yolo((w, h), box)
        boxes.append(yolo_box)
        labels.append(class_id)
    return boxes, labels, w, h

# Save YOLO labels
def save_yolo_label(path, bboxes, labels):
    with open(path, 'w') as f:
        for label, box in zip(labels, bboxes):
            f.write(f"{label} {' '.join(map(str, box))}\n")

# Main loop over train/val/test
for split in SPLITS:
    img_dir = os.path.join(INPUT_ROOT, split)
    all_images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    for img_path in tqdm(all_images, desc=f"Processing {split}"):
        filename = os.path.basename(img_path)
        xml_path = Path(img_path).with_suffix('.xml')

        if not os.path.exists(xml_path):
            print(f"XML not found for: {filename}")
            continue

        bboxes, labels, w, h = parse_voc_xml(xml_path)
        if not bboxes:
            print(f"No relevant labels in: {filename}")
            continue

        image = cv2.imread(img_path)

        # Save original image and label
        out_img_path = f"{OUTPUT_ROOT}/{split}/images/{filename}"
        out_lbl_path = f"{OUTPUT_ROOT}/{split}/labels/{filename.replace('.jpg', '.txt')}"
        shutil.copy(img_path, out_img_path)
        save_yolo_label(out_lbl_path, bboxes, labels)

        # Augmentations (2 copies per image)
        for i in range(2):
            aug = augmenter(image=image, bboxes=bboxes, class_labels=labels)
            aug_img = aug['image']
            aug_boxes = aug['bboxes']
            aug_labels = aug['class_labels']
            aug_fname = filename.replace('.jpg', f'_aug{i}.jpg')

            cv2.imwrite(f"{OUTPUT_ROOT}/{split}/images/{aug_fname}", aug_img)
            save_yolo_label(f"{OUTPUT_ROOT}/{split}/labels/{aug_fname.replace('.jpg', '.txt')}", aug_boxes, aug_labels)

print("\n Preprocessing and augmentation (original + augmented) complete.")


# # 📂 YOLOv8 Parking Dataset Summary
# 
# ## 🔗 Dataset Download Link
# You can download the full YOLOv8-compatible dataset (including images, labels, and `data.yaml`) from the following Google Drive link:
# 
# **[Download Dataset from Google Drive](https://drive.google.com/drive/folders/13KoJEiEwVik8_quz2z4oE8DF1LfieQ3f?usp=sharing)**  
# *(Make sure the link has proper sharing permissions)*
# 
# ## 📊 Dataset Split Summary
# | Split       | Number of Samples |
# |-------------|-------------------|
# | Train       | 25,507            |
# | Validation  | 7,273             |
# | Test        | 3,649             |
# | **Total**   | **36,429**        |
# 
# ## ✅ Contents
# - YOLOv8-compatible structure with `images/` and `labels/` folders.
# - Augmented versions of images are included.
# - `data.yaml` configured for 2 classes:
#   - `0`: `space-empty`
#   - `1`: `space-occupied`
# 

# # 📄 Generate `data.yaml` for YOLOv8
# 
# ## 🗂 1. Dataset Root
# - Set the root directory where the YOLO-formatted dataset is saved.
# 
# ## 📋 2. Define `data.yaml` Structure
# - YAML structure includes:
#   - `train`: Path to training images.
#   - `val`: Path to validation images.
#   - `test`: Path to test images.
#   - `nc`: Number of classes (2 in this case).
#   - `names`: Class names in the correct order for YOLO:
#     - `0` -> `space-empty`
#     - `1` -> `space-occupied`
# 
# ## 💾 3. Save YAML File
# - The YAML dictionary is saved to `data.yaml` in the dataset root.
# - This file is required to train YOLOv8 using the Ultralytics framework.
# 
# ## ✅ 4. Output
# - A `data.yaml` file is generated at:
# 
# 

# In[5]:


import yaml
from pathlib import Path

# Define dataset root path
dataset_root = Path("/home/user/YOLO_Assignment/Task_1/Dataset/dataset_yolo")

# Define YAML content
data = {
    'train': str(dataset_root / 'train' / 'images'),
    'val': str(dataset_root / 'valid' / 'images'),
    'test': str(dataset_root / 'test' / 'images'),
    'nc': 2,
    'names': ['space-empty', 'space-occupied']
}

# Save YAML
yaml_path = dataset_root / "data.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(data, f)

print(f" data.yaml generated at: {yaml_path}")


# In[ ]:




