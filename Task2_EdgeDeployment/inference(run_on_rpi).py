#!/usr/bin/env python
# coding: utf-8

# ### 🧠 Inference Script for Parking Slot Detection using ONNX + OpenCV
# 
# This script performs the following steps:
# 
# 1. **Model and Image Setup**
#    - Loads a YOLOv8 ONNX model from a specified path.
#    - Reads and resizes a test image to `640x640` for inference.
# 
# 2. **Dual Logging Setup**
#    - All outputs are logged both to the terminal and a log file named `inference_log.txt`.
# 
# 3. **ONNX Inference**
#    - Runs the model on the image using `onnxruntime`.
#    - Measures and logs FPS (Frames Per Second), CPU usage, and GPU memory usage using `psutil` and `GPUtil`.
# 
# 4. **Postprocessing**
#    - Parses the output to extract bounding boxes, confidence scores, and class IDs.
#    - Scales the boxes back to the original image size.
#    - Filters predictions based on a confidence threshold (`CONF_THRESH = 0.25`).
# 
# 5. **Visualization**
#    - Draws bounding boxes and class labels (`space-empty` or `space-occupied`) on the original image.
#    - Saves the annotated image as `output.jpg`.
# 
# 6. **Slot Counting**
#    - Counts the number of detected empty and occupied parking slots.
#    - Displays the counts and logs them.
# 
# 7. **Log File Output**
#    - All printed information is saved to `inference_log.txt`.
# 
# This script is useful for testing YOLOv8 ONNX models and logging system performance metrics during inference. It helps in evaluating edge deployment readiness and detection quality.
# 

# In[1]:


import onnxruntime as ort
import cv2
import numpy as np
import time
import psutil
import GPUtil
import os
import sys

# === CONFIG ===
ONNX_PATH = "/home/user/YOLO_Assignment/Task_1/Model/runs_train_val/detect/yolov8_parking/weights/best.onnx"
IMG_PATH = "/home/user/YOLO_Assignment/Task_2/sample_inputs/2012-12-22_13_20_09_jpg.rf.8fa5f4f25da5d974c608cbe84afbc9e6.jpg"
CLASS_NAMES = ['space-empty', 'space-occupied']
CONF_THRESH = 0.25
LOG_FILE = "inference_log.txt"

# === SETUP DUAL LOGGING ===
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.__stdout__
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_FILE)

# === BEGIN ===
print("🚀 Starting Inference\n")
print(f"Model: {ONNX_PATH}")
print(f"Image: {IMG_PATH}")

# === LOAD MODEL ===
assert os.path.exists(ONNX_PATH), f"Model not found: {ONNX_PATH}"
session = ort.InferenceSession(ONNX_PATH)

# === PREPROCESS IMAGE ===
orig_img = cv2.imread(IMG_PATH)
resized_img = cv2.resize(orig_img, (640, 640))
img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB).astype(np.float32)
img_rgb /= 255.0
img_rgb = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, ...]

# === INFERENCE ===
start = time.time()
outputs = session.run(None, {"images": img_rgb.astype(np.float32)})
end = time.time()

# === PERFORMANCE ===
fps = 1 / (end - start)
print(f"\n FPS: {fps:.2f}")
print(f"CPU Usage: {psutil.cpu_percent()}%")
for gpu in GPUtil.getGPUs():
    print(f"GPU: {gpu.name}, Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

# === POSTPROCESS ===
detections = outputs[0][0]
boxes = []
slot_counts = {'space-empty': 0, 'space-occupied': 0}

for det in detections:
    x1, y1, x2, y2, conf, cls = det
    if conf < CONF_THRESH:
        continue
    x1 = int(x1 * orig_img.shape[1] / 640)
    y1 = int(y1 * orig_img.shape[0] / 640)
    x2 = int(x2 * orig_img.shape[1] / 640)
    y2 = int(y2 * orig_img.shape[0] / 640)
    cls_id = int(cls)
    boxes.append((x1, y1, x2, y2, float(conf), cls_id))
    slot_counts[CLASS_NAMES[cls_id]] += 1

# === DRAW ===
for x1, y1, x2, y2, conf, cls_id in boxes:
    label = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(orig_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("output.jpg", orig_img)
print("\n Saved annotated output image as 'output.jpg'")

# === COUNTS ===
print("\n Parking Slot Counts:")
print(f"  Empty Slots    : {slot_counts['space-empty']}")
print(f"  Occupied Slots : {slot_counts['space-occupied']}")

# === CLOSE LOG ===
sys.stdout.log.close()
sys.stdout = sys.__stdout__
print("\n All done. Log saved to inference_log.txt")


# ### 📁 Output Resources on Google Drive
# 
# Below are the links to various output folders stored on Google Drive:
# 
# - 📄 **Inference Logs**:  
#   Contains system logs, FPS, CPU/GPU usage, and slot counts per image/video.  
#   🔗 [Open inference_logs folder](https://drive.google.com/drive/folders/1od03HzUF7iAwpSf-_z77dke_-ShzL_yY?usp=sharing)
# 
# - 🎥 **Sample Video Output**:  
#   Contains video clips with annotated bounding boxes for occupied and empty parking slots.  
#   🔗 [Open sample_video_output folder](https://drive.google.com/drive/folders/1VpYMTd6B6tLJEJUlPzjvI9kebkD85RmC?usp=sharing)
# 
# - 🖼️ **Sample Image Output**:  
#   Contains processed images with bounding boxes and class labels.  
#   🔗 [Open sample_image_output folder](https://drive.google.com/drive/folders/1xYDqrI5ZoaziP1mYRBHt8_0OPY6_Q2CZ?usp=sharing)
# 

# ### ✅ Inference Results Summary
# 
# **🖼️ Input Image**:  
# `2012-12-22_13_20_09_jpg.rf.8fa5f4f25da5d974c608cbe84afbc9e6.jpg`  
# 
# **💾 Output**:  
# Annotated image saved as: `output.jpg`
# 
# ---
# 
# **📊 Detection Summary**:
# - **Empty Slots**: `26`
# - **Occupied Slots**: `1`
# 
# ---
# 
# **🖥️ System Performance**:
# - **FPS (Frames Per Second)**: `13.47`
# - **CPU Usage**: `0.2%`
# - **GPU**: Quadro RTX 5000  
#   → **Memory Used**: `46.0MB` / `16384.0MB`
# 
# ---
# 
# This result confirms successful inference on the uploaded parking lot image with low resource usage and accurate slot classification.
# 

# In[ ]:




