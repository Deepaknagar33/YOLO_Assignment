#!/usr/bin/env python
# coding: utf-8

# ### 🔍 YOLOv8 Model Validation & GPU Memory Usage
# 
# This script performs the following key steps for validating a trained YOLOv8 model:
# 
# 1. **Imports Required Libraries**:
#    - `ultralytics.YOLO` to load and validate the model.
#    - `torch` to handle GPU operations and memory usage.
# 
# 2. **Configuration**:
#    - Sets the model path:  
#      `/home/user/YOLO_Assignment/Task_1/Model/runs/detect/yolov8_parking/weights/best.pt`
#    - Checks for GPU availability and sets the device accordingly.
# 
# 3. **Model Loading**:
#    ```python
#    model = YOLO(MODEL_PATH)
# 

# In[2]:


from ultralytics import YOLO
import torch

# === CONFIGURATION ===
MODEL_PATH = '/home/user/YOLO_Assignment/Task_1/Model/runs/detect/yolov8_parking/weights/best.pt'
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === VALIDATE ===
results = model.val()
print(" Validation Done. Results:")
print(results)

# === GPU MEMORY ===
if torch.cuda.is_available():
    mem = torch.cuda.memory_allocated(DEVICE) / (1024 ** 3)
    print(f" GPU Memory Used: {mem:.2f} GB")

# === MODEL INFO ===
model.info(verbose=True)


# ### ✅ TASK-1 Validation Results
# 
# **Dataset Summary:**
# - **Total Images:** 7,272  
# - **Total Instances:** 429,948  
# 
# ---
# 
# **Class-wise Performance Metrics:**
# 
# | Class             | Images | Instances | Precision (P) | Recall (R) | mAP@0.5 | mAP@0.5:0.95 |
# |------------------|--------|-----------|---------------|------------|--------|--------------|
# | **All Classes**        | 7,272  | 429,948   | 0.999         | 0.996      | 0.995  | 0.982        |
# | **space-empty**        | 6,186  | 220,887   | 0.999         | 0.997      | 0.995  | 0.985        |
# | **space-occupied**     | 5,901  | 209,061   | 0.998         | 0.995      | 0.994  | 0.980        |
# 
# ---
# 
# **Inference Speed:**
# - Preprocessing: **0.1 ms/image**
# - Inference: **2.9 ms/image**
# - Postprocessing: **1.9 ms/image**
# 
# **Loss Calculation:** 0.0 ms/image  
# **GPU Memory Used:** 0.01 GB
# 

# ---
# 
# # Confusion Matrix Summary
# 
# 
# 
# The confusion matrix obtained from the validation data is shown below:
# 
# | Predicted \ Actual | space-empty | space-occupied | background |
# |--------------------|-------------|----------------|------------|
# | **space-empty**    | 220,442     | 240            | 515        |
# | **space-occupied** | 296         | 208,800        | 334        |
# 
# 
# > ✅ **Note**: The diagonal values represent correct predictions for each class.
# 
# ---
# 
# ## 3. Key Observations
# 
# ### Correct Predictions:
# - **space-empty**: 220,442
# - **space-occupied**: 208,800
# 
# 
# ---
# 
# ## 4. Performance Summary
# 
# ### Accuracy:
# $$
# \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{220,442 + 208,800 + 0}{220,442 + 240 + 515 + 296 + 208,800 + 334 + 149 + 21 + 0} = \frac{429,242}{430,797} \approx 0.996
# $$
# 
# - **Accuracy ≈ 99.6%**
# 
# 
# 
# ---
# 
# ## 5. Validation Results
# 
# 📥 [Download or View Full Validation Results](https://drive.google.com/drive/folders/18yyL4Iie--TlSIPHa5jpLHc81pMIiMg7?usp=sharing) (Google Drive Link)
# 
# 
# 
# ---

# In[ ]:




