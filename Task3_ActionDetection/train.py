#!/usr/bin/env python
# coding: utf-8

# # YOLOv8 Training Configuration
# 
# ## Model Setup
# ```python
# model = YOLO('yolov8n.pt')  # Load nano variant

# In[1]:


from ultralytics import YOLO

# Load YOLOv8n model (nano version)
model = YOLO('yolov8n.pt')

# Train on your custom dataset
model.train(
    data='/home/user/YOLO_Assignment/Task_3/yolo_dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='ActionYOLO',
    name='dribble_fall',
    verbose=True
)


# 
# # GPU Utilization Summary (Training for 50 Epochs)
# 
# ## Overview
# 
# This section provides a summary of the **GPU memory utilization** during the training of the model over **50 epochs**.
# 
# The GPU used had consistent memory usage starting from **epoch 4 onwards**, with slight variation in the initial epochs.
# 
# ---
# 
# ## GPU Memory Usage Per Epoch
# 
# | Epoch | GPU Memory Usage |
# |-------|------------------|
# | 1     | 2.09 GB          |
# | 2     | 2.66 GB          |
# | 3     | 2.66 GB          |
# | 4 to 50 | 2.66 GB (constant) |
# 
# 

# 
# ## 📁 Training Results
# 
# 📥 You can access the complete training results, including logs, metrics, and model checkpoints, via the following Google Drive link:
# 
# 👉 [Download/View Training Results on Google Drive](https://drive.google.com/drive/folders/1ty1iAffsr0Y9p8VJv2a80YCbrEsVR65E?usp=sharing)
# 
# > 🔐 Make sure the folder is shared with "Anyone with the link" in view/download mode for seamless access.
# 
# 

# 
# ## 📊 Results Summary (Excel File)
# 
# 📄 You can access the **Excel file containing detailed training/validation results** (accuracy, loss, metrics per epoch, etc.) using the link below:
# 
# 👉 [Download/View Results Excel File (Google Drive)](https://drive.google.com/file/d/1teEbX5rSgzU4Tu9xHFulQcyK_ZkAit8h/view?usp=sharing)
# 
# > ✅ This `.xlsx` file includes multiple sheets with:
# > - Training and validation metrics per epoch  
# > - Confusion matrix  
# > - Precision, Recall, F1-Score for each class  
# > - Loss and accuracy graphs  
# 
# > 🔐 Make sure you have access permissions set to "Anyone with the link" for smooth access.
# 

# In[ ]:




