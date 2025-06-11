#!/usr/bin/env python
# coding: utf-8

# # YOLO Dataset Preparation Pipeline
# 
# ## Overview
# This script converts extracted video frames into a YOLO-formatted dataset with automatic full-frame bounding box annotations. Due to time constraints, we implemented this automated approach instead of manual labeling with the use of LabelImg.
# 
# ## Dataset Structure
# 
# ```python
# yolo_dataset/
# ├── images/
# │   ├── train/      # Training images (80%)
# │   └── val/        # Validation images (20%)
# └── labels/
#     ├── train/      # Corresponding label files
#     └── val/

# In[1]:


import os
import cv2
import random
from glob import glob
from shutil import copy2

# Paths
input_root = "/home/user/YOLO_Assignment/Task_3/processed_frames"
output_root = "/home/user/YOLO_Assignment/Task_3/yolo_dataset"
os.makedirs(output_root, exist_ok=True)

# Class to ID mapping
classes = ['dribble', 'fall_floor']
class_map = {cls: idx for idx, cls in enumerate(classes)}

# Create YOLO directories
for split in ['train', 'val']:
    os.makedirs(f"{output_root}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_root}/labels/{split}", exist_ok=True)

# Helper: Convert full image box to YOLO format
def full_frame_box(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    x_center, y_center = 0.5, 0.5
    width, height = 1.0, 1.0
    return x_center, y_center, width, height

# Process each class folder
for cls in classes:
    all_imgs = glob(f"{input_root}/{cls}/*.jpg")
    random.shuffle(all_imgs)
    split_idx = int(0.8 * len(all_imgs))
    splits = {'train': all_imgs[:split_idx], 'val': all_imgs[split_idx:]}

    for split, files in splits.items():
        for img_path in files:
            img_name = os.path.basename(img_path)
            label_name = img_name.replace('.jpg', '.txt')

            # Copy image
            copy2(img_path, f"{output_root}/images/{split}/{img_name}")

            # Create full-frame box annotation
            x, y, w, h = full_frame_box(img_path)
            with open(f"{output_root}/labels/{split}/{label_name}", 'w') as f:
                f.write(f"{class_map[cls]} {x} {y} {w} {h}\n")

print("✅ Labeling complete. YOLO dataset ready.")


# # YOLO Dataset Configuration File Generator
# 
# ## Overview
# This script generates a `data.yaml` configuration file for YOLO (v5/v8) datasets, which specifies:
# - Dataset directory structure
# - Training/validation paths
# - Number of classes
# - Class names
# 
# ## Generated YAML Structure
# 
# ```yaml
# path: /home/user/YOLO_Assignment/Task_3/yolo_dataset
# train: images/train
# val: images/val  
# nc: 2
# names: ['dribble', 'fall_floor']

# In[2]:


import yaml
import os

# Define dataset structure
dataset_path = "/home/user/YOLO_Assignment/Task_3/yolo_dataset"
yaml_dict = {
    "path": dataset_path,
    "train": "images/train",
    "val": "images/val",
    "nc": 2,
    "names": ["dribble", "fall_floor"]
}

# Save as data.yaml
yaml_file = os.path.join(dataset_path, "data.yaml")
with open(yaml_file, 'w') as f:
    yaml.dump(yaml_dict, f)

print(f"✅ data.yaml created at: {yaml_file}")


#  YOLO Formatted Dataset with Annotations
# 
# ## 📁 Dataset Download
# [![Google Drive](https://img.shields.io/badge/Google%20Drive-Dataset_With_Annotations-4285F4?style=for-the-badge&logo=googledrive)](https://drive.google.com/drive/folders/1lJK_GmlZ9k1v-GJdZT1jxIxld4-NaMBa?usp=sharing)

# In[ ]:




