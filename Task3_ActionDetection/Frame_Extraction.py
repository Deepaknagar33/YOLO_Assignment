#!/usr/bin/env python
# coding: utf-8

# # Video Frame Extraction Pipeline
# 
# ## Overview
# This Python script processes the HMDB51 action recognition dataset to:
# 1. Extract RAR archives containing video files
# 2. Sample frames from videos at a specified frame rate
# 3. Save frames as JPEG images for downstream tasks
# 
# ## Configuration
# 
# ```python
# # Dataset paths
# base_dir = '/home/user/YOLO_Assignment/Task_3/DataSet/HMDB51/hmdb51_org'
# output_dir = '/home/user/YOLO_Assignment/Task_3/processed_frames'
# 
# # Target action classes to process
# target_classes = ['fall_floor', 'dribble']
# 
# # Frame extraction rate (1 frame per second)
# frame_rate = 1  

# In[2]:


import os
import cv2
import rarfile

# Set paths
base_dir = '/home/user/YOLO_Assignment/Task_3/DataSet/HMDB51/hmdb51_org'
output_dir = '/home/user/YOLO_Assignment/Task_3/processed_frames'
target_classes = ['fall_floor', 'dribble']
frame_rate = 1  # 1 frame per second

# Step 1: Extract RAR files
for cls in target_classes:
    rar_path = os.path.join(base_dir, f"{cls}.rar")
    extract_path = os.path.join(base_dir, cls)
    if not os.path.exists(extract_path):
        print(f"Extracting {cls}.rar...")
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(path=extract_path)

# Step 2: Extract frames from videos
for cls in target_classes:
    video_dir = os.path.join(base_dir, cls)
    out_cls_dir = os.path.join(output_dir, cls)
    os.makedirs(out_cls_dir, exist_ok=True)

    for vid_file in os.listdir(video_dir):
        if not vid_file.endswith('.avi'):
            continue
        video_path = os.path.join(video_dir, vid_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps // frame_rate)

        frame_idx = 0
        saved_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frame_name = f"{os.path.splitext(vid_file)[0]}_frame{saved_idx}.jpg"
                frame_path = os.path.join(out_cls_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                saved_idx += 1
            frame_idx += 1
        cap.release()
        print(f"Extracted frames from {vid_file} to {out_cls_dir}")


# # HMDB51 Frame Extraction Results
# 
# ## Dataset Overview
# - **Source Dataset**: HMDB51 Action Recognition
# - **Target Classes Processed**: `dribble`, `fall_floor`
# - **Total Videos Processed**: 283
# - **Total Frames Extracted**: 841
# 
# ## 📁 Download Extracted Frames
# [![Google Drive](https://img.shields.io/badge/Google%20Drive-Extracted_Frames-4285F4?style=for-the-badge&logo=googledrive)](https://drive.google.com/drive/folders/1DgYXhJnSsyfAquxXY6MOJTNxBWKH08Nj?usp=sharing)
# 
# *Folder contains organized frames in class-specific subdirectories*
# 
# ## Class-wise Statistics
# 
# ### 🏀 Dribble Class
# | Metric | Count |
# |--------|-------|
# | Videos | 146 |
# | Frames | 528 |
# | Avg Frames/Video | 3.62 |
# 
# ### 🩺 Fall_Floor Class
# | Metric | Count |
# |--------|-------|
# | Videos | 137 |
# | Frames | 313 |
# | Avg Frames/Video | 2.28 |
# 
# ## Extraction Parameters
# ```python
# {
#     "frame_rate": "1 fps",
#     "format": "JPEG",
#     "resolution": "Original video resolution",
#     "naming_convention": "[video_name]_frame[number].jpg"
# }

# In[ ]:




