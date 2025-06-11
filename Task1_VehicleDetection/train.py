#!/usr/bin/env python
# coding: utf-8

# # 🚀 YOLOv8 Model Training on Parking Dataset
# 
# ## ⚙️ Configuration
# - **Model Used**: `yolov8n.pt` (YOLOv8 Nano – lightweight and fast)
# - **Dataset YAML**:  
#   `/home/user/YOLO_Assignment/Task_1/Dataset/dataset_yolo/data.yaml`
# - **Training Device**: GPU if available, else CPU
# - **Project Name**: `yolov8_parking`
# 
# ## 🧠 Model Loading
# - Loads the pretrained YOLOv8 model using Ultralytics' API.
# 
# ## 🏋️ Training Parameters
# - `epochs = 50`
# - `image size = 640 × 640`
# - `batch size = 48`
# - `workers = 2` (for data loading)
# - Trains on the specified dataset and saves results under `runs/detect/yolov8_parking/`.
# 
# ## 🧾 Model Summary
# - After training, the script prints a detailed summary of the model architecture, layers, and parameter count using `model.info()`.
# 
# ## 💻 GPU Usage Monitoring
# - If using a GPU, the script reports the total memory used after training
# 

# In[1]:


from ultralytics import YOLO
import torch

# === CONFIGURATION ===
DATA_YAML = '/home/user/YOLO_Assignment/Task_1/Dataset/dataset_yolo/data.yaml'
MODEL_NAME = 'yolov8n.pt'
PROJECT_NAME = 'yolov8_parking'
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# === LOAD MODEL ===
model = YOLO(MODEL_NAME)

# === TRAIN ===
model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=48,
    name=PROJECT_NAME,
    workers=2,
    device=DEVICE
)

# === MODEL INFO ===
model.info(verbose=True)

# === GPU MEMORY ===
if torch.cuda.is_available():
    mem = torch.cuda.memory_allocated(DEVICE) / (1024 ** 3)
    print(f" GPU Memory Used: {mem:.2f} GB")



# # 📊 YOLOv8 Training Results on Parking Slot Detection
# 
# ## 🔍 1. Performance Overview
# 
# The model was trained for **50 epochs** and demonstrates strong convergence and generalization. Below is a summary based on the plots:
# 
# - **Loss Curves**:
#   - All training and validation losses (`box_loss`, `cls_loss`, `dfl_loss`) decreased steadily.
#   - Final validation losses indicate low overfitting and effective learning.
# 
# - **Metric Trends**:
#   - Precision and recall quickly reached and stabilized near **1.0** for both classes.
#   - **mAP@0.5** ≈ 0.995 and **mAP@0.5:0.95** ≈ 0.96 by the end of training, confirming excellent object localization and classification.
# 
# - **Confidence Curves**:
#   - The **Precision-Confidence** and **Recall-Confidence** curves show high confidence even for borderline predictions.
#   - The **F1-Confidence** curve peaked at 1.00 around a confidence threshold of **0.49**, ideal for detection threshold tuning.
# 
# - **Precision-Recall Curve**:
#   - Both classes show very high AP scores (`space-empty`: 0.995, `space-occupied`: 0.994), indicating well-balanced precision and recall.
# 
# - **Confusion Matrix**:
#   - Extremely low misclassifications.
#   - Classifier learned to distinguish `space-empty` and `space-occupied` effectively.
#   - Background class errors are negligible.
# 
# ---
# 
# ## 📁 2. Downloadable Results
# 
# - 📥 **Excel Sheet (Per-Epoch Metrics)**  
#   [Download Epoch-wise Metrics (Excel)](https://drive.google.com/file/d/1LDjOU6wxvAFWgs2uyUWKnmhUJ7GEgqrN/view?usp=sharing)
# 
# - 📂 **Full Results Folder (YOLO Output)**  
#   [Access Google Drive Folder with Model Weights, Plots, Predictions](https://drive.google.com/drive/folders/1qZ4q-siUfPXLwo3a321fQ47JmkkRv7NR?usp=sharing)
# 
# > ⚠️ *Ensure that the Google Drive links are publicly accessible (Anyone with the link can view).*
# 
# ---
# 
# ## ✅ 3. Conclusion
# 
# The YOLOv8n model trained on the parking dataset achieved **state-of-the-art performance** with:
# - High accuracy and minimal loss.
# - Robust classification of both `space-empty` and `space-occupied` classes.
# - Suitable for deployment in real-world automated parking management systems.
# 
# 

# ### GPU Utilization Per Epoch (1 to 50)
# 
# The following table shows the GPU memory usage (in GB) during training across 50 epochs.
# 
# | Epoch | GPU Utilization (GB) | Epoch | GPU Utilization (GB) |
# |-------|----------------------|-------|----------------------|
# | 1     | 12.2                 | 26    | 15.1                 |
# | 2     | 11.3                 | 27    | 11.2                 |
# | 3     | 11.7                 | 28    | 13.3                 |
# | 4     | 15.4                 | 29    | 12.3                 |
# | 5     | 11.7                 | 30    | 15.1                 |
# | 6     | 13.4                 | 31    | 11.4                 |
# | 7     | 15.0                 | 32    | 13.6                 |
# | 8     | 14.0                 | 33    | 15.3                 |
# | 9     | 13.8                 | 34    | 11.1                 |
# | 10    | 13.8                 | 35    | 13.3                 |
# | 11    | 13.7                 | 36    | 14.8                 |
# | 12    | 10.9                 | 37    | 12.8                 |
# | 13    | 11.2                 | 38    | 11.8                 |
# | 14    | 11.4                 | 39    | 13.1                 |
# | 15    | 12.8                 | 40    | 10.4                 |
# | 16    | 13.1                 | 41    | 6.78                 |
# | 17    | 12.6                 | 42    | 7.53                 |
# | 18    | 13.4                 | 43    | 7.53                 |
# | 19    | 10.9                 | 44    | 7.53                 |
# | 20    | 14.9                 | 45    | 7.53                 |
# | 21    | 10.8                 | 46    | 7.53                 |
# | 22    | 11.1                 | 47    | 7.53                 |
# | 23    | 11.9                 | 48    | 7.53                 |
# | 24    | 13.4                 | 49    | 7.53                 |
# | 25    | 15.1                 | 50    | 7.53                 |
# 
# 
# 

# In[ ]:




