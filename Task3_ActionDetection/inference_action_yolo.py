#!/usr/bin/env python
# coding: utf-8

# # YOLOv8 Inference Notebook
# 
# ## 📌 Overview
# This notebook demonstrates how to perform inference using a trained YOLOv8 model on a video file, with GPU monitoring and performance logging.
# 
# ```python
# # Import required libraries
# from ultralytics import YOLO
# import torch
# import GPUtil
# import psutil
# import time
# from datetime import timedelta
# import sys
# import os
# from glob import glob
# import shutil
# 
# # Setup logging
# log_dir = "logs"
# os.makedirs(log_dir, exist_ok=True)
# log_file_path = os.path.join(log_dir, "inference_log.txt")
# sys.stdout = open(log_file_path, "w")
# 
# print("🚀 Starting Model Inference...\n")

# In[1]:


from ultralytics import YOLO
import torch
import GPUtil
import psutil
import time
from datetime import timedelta
import sys
import os
from glob import glob
import shutil

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "inference_log.txt")
sys.stdout = open(log_file_path, "w")

print("🚀 Starting Model Inference...\n")

# Device Info
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device.upper()}\n")

# Get GPU usage before inference
gpus_before = GPUtil.getGPUs()
print("📊 GPU Usage Before Inference:")
for gpu in gpus_before:
    print(f"  Name: {gpu.name}, Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

# Start timer
start_time = time.time()

# Load trained model
model = YOLO('ActionYOLO/dribble_fall/weights/best.pt')

# Path to test video
test_video_path = '/home/user/YOLO_Assignment/Task_3/DataSet/HMDB51/hmdb51_org/fall_floor/Pirates_7_fall_floor_f_nm_np3_le_med_2.avi'

print(f"\n🎬 Running inference on video: {test_video_path}")

# Run prediction
results = model.predict(
    source=test_video_path,
    save=True,
    save_txt=True,
    save_conf=True,
    project='ActionYOLO',
    name='demo_inference',
    conf=0.25
)

# End timer
end_time = time.time()
elapsed = end_time - start_time
print(f"\n⏱️ Inference completed in: {str(timedelta(seconds=int(elapsed)))}")

# Get GPU usage after inference
gpus_after = GPUtil.getGPUs()
print("\n📊 GPU Usage After Inference:")
for gpu in gpus_after:
    print(f"  Name: {gpu.name}, Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

# System Info
print("\n🖥️ System Info:")
print(f"  CPU: {psutil.cpu_percent(interval=1)}% used")
print(f"  RAM: {psutil.virtual_memory().percent}% used")

# Save final video
inference_dir = 'ActionYOLO/demo_inference'
video_candidates = glob(os.path.join(inference_dir, '*.avi')) + glob(os.path.join(inference_dir, '*.mp4'))

if video_candidates:
    source_video = video_candidates[0]
    destination_video = 'ActionYOLO/final_demo_summary.mp4'
    shutil.copyfile(source_video, destination_video)
    print(f"\n✅ Predicted video saved for summary as: {destination_video}")
else:
    print("\n❌ No predicted video found in inference directory.")

# Restore stdout
sys.stdout.close()
sys.stdout = sys.__stdout__

print(f"\n💾 Inference log saved to: {log_file_path}")


# ## 📂 Sample Outputs
# 
# Here are the sample outputs from the inference process:
# 
# ### 📁 Predicted Videos Folder
# 🔗 [Open Videos Folder](https://drive.google.com/drive/folders/1df38dCqChDM-ZdEALYAGiZhP7sXHkJDS?usp=sharing)
# 
# ### 📄 Inference Log File
# 📝 [View Inference Log](https://drive.google.com/file/d/1TlrkBfI7qHacnjOgbVUWdZCh-s8C0ZSN/view?usp=sharing)  
# 🔍 Contains detailed timing, hardware usage, and process information
# 
# ### 📊 Inference Result Data
# 🗃️ [Download Result Data](https://drive.google.com/drive/folders/1Pz9us3Gw4dgESLv_1Fjrfpf-8ma1VHIc?usp=sharing)  
# Includes:
# - Detection labels (YOLO format)
# - Confidence scores
# - Bounding box coordinates
# - Frame-by-frame analysis
# 
# 

# In[ ]:




