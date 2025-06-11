#!/usr/bin/env python
# coding: utf-8

# 
# 
# # 📊 Model Validation Report with GPU Utilization
# 
# ## 🔧 Script Overview
# 
# This script performs **model validation** using a trained YOLO model (`best.pt`) and logs detailed information including:
# 
# - Device used (CPU/GPU)
# - GPU memory usage before and after validation
# - Total validation time
# - Evaluation metrics (mAP, precision, recall, etc.)
# - System resource usage (CPU & RAM)
# 
# All output is saved to a log file for future reference.
# 
# ---
# 
# ## 🧠 Model Details
# 
# - **Model Path**: `ActionYOLO/dribble_fall/weights/best.pt`
# - **Validation Command**: `model.val()`
# - **Classes Evaluated**:
#   - `dribble`
#   - `fall_floor`
#   - `all` (combined metrics)
# 
# ---
# 
# ## 🖥️ Execution Environment
# 
# | Component | Value |
# |----------|-------|
# | **Device** | CUDA (if available) or CPU |
# | **Logging** | Output redirected to `logs/validation_log.txt` |
# | **GPU Monitoring** | GPUtil |
# | **System Monitoring** | psutil |
# 
# ---
# 
# ## 📈 Key Metrics Logged
# 
# ### ⏱️ Performance Timing
# - Start time: `time.time()`
# - End time: `time.time()`
# - Total elapsed time: displayed in `HH:MM:SS` format
# 
# ### 📊 Evaluation Metrics
# The following are printed and logged from `metrics` object:
# - **Precision (P)**
# - **Recall (R)**
# - **mAP@0.5**
# - **mAP@0.5:0.95**
# 
# 
# 
# ## 📁 Output Files
# 
# | File | Description |
# |------|-------------|
# | `logs/validation_log.txt` | Full log of validation process including GPU/CPU/RAM usage and evaluation metrics |
# | `runs/detect/val` | Directory where prediction results are saved during validation |
# 
# 

# In[2]:


from ultralytics import YOLO
import torch
import GPUtil
import psutil
import time
from datetime import timedelta
import sys
import os

# Redirect stdout to log file
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "validation_log.txt")
sys.stdout = open(log_file_path, "w")

print("🚀 Starting Model Validation...\n")

# Load trained model
model = YOLO('ActionYOLO/dribble_fall/weights/best.pt')

# Get device info
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device.upper()}\n")

# Start time
start_time = time.time()

# Track GPU before inference
gpus_before = GPUtil.getGPUs()
print("📊 GPU Usage Before Validation:")
for gpu in gpus_before:
    print(f"  Name: {gpu.name}, Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

# Evaluate on validation set
print("\n🔍 Running Validation...\n")
metrics = model.val()
print("\n✅ Evaluation Metrics:")
print(metrics)

# End time
end_time = time.time()
elapsed = end_time - start_time
print(f"\n⏱️ Total Validation Time: {str(timedelta(seconds=int(elapsed)))}")

# Track GPU after inference
gpus_after = GPUtil.getGPUs()
print("\n📊 GPU Usage After Validation:")
for gpu in gpus_after:
    print(f"  Name: {gpu.name}, Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

# System Info
print("\n🖥️ System Info:")
print(f"  CPU: {psutil.cpu_percent(interval=1)}% used")
print(f"  RAM: {psutil.virtual_memory().percent}% used")

# Restore stdout
sys.stdout.close()
sys.stdout = sys.__stdout__

print(f"\n💾 Validation log saved to: {log_file_path}")


# 
# 
# ### 📁 Validation Results & Logs
# 
# 📥 You can access the following resources from Google Drive:
# 
# 🔗 **Download/View Full Validation Results (metrics, output files, etc.)**  
# 👉 [Validation Results Folder](https://drive.google.com/drive/folders/1sBAIgfWFD2Qdz6jTcwKDzDs6gya4HTrb?usp=sharing)
# 
# 📄 **Download Validation Log File (`validation_log.txt`)**  
# 👉 [Download validation_log.txt](https://drive.google.com/file/d/1jsLgjwdt6YIY6k1F_qB0K377BvLXc3Em/view?usp=sharing)
# 
# > 🔐 Make sure both the folder and file are shared with "Anyone with the link" for seamless access.
# 
# ---
# 
# Let me know if you'd like this styled as part of a report or notebook!

# In[ ]:




