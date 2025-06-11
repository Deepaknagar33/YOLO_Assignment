#!/usr/bin/env python
# coding: utf-8

# ### 🔄 Exporting YOLOv8 Model to ONNX Format
# 
# The following code exports a trained YOLOv8 model (`best.pt`) to ONNX format with proper postprocessing enabled (NMS, confidence thresholding, and class decoding):
# 
# 
# 

# In[15]:


from ultralytics import YOLO

model = YOLO("/home/user/YOLO_Assignment/Task_1/Model/runs_train_val/detect/yolov8_parking/weights/best.pt")

#  Important: Enable postprocessing by setting `nms=True` (enabled by default from v8.1)
model.export(format="onnx", imgsz=640, dynamic=True, simplify=True, opset=12,nms=True)


# ### ✅ ONNX Model Validation
# 
# The following code is used to **validate the structure and integrity** of the exported YOLOv8 ONNX model:
# 
# ```python
# import onnx
# 
# # Load the model from disk
# model = onnx.load('/home/user/YOLO_Assignment/Task_1/Model/runs_train_val/detect/yolov8_parking/weights/best.onnx')
# 
# # Run structural checks on the model
# onnx.checker.check_model(model)
# 
# print("✅ ONNX model is valid!")
# 

# In[3]:


import onnx

# Load the model
model = onnx.load('/home/user/YOLO_Assignment/Task_1/Model/runs_train_val/detect/yolov8_parking/weights/best.onnx')

# Check if the model is valid
onnx.checker.check_model(model)
print(" ONNX model is valid!")


# ### ⚙️ Dynamic Quantization of YOLOv8 ONNX Model
# 
# The following code applies **dynamic quantization** to the exported YOLOv8 ONNX model to reduce model size and improve inference efficiency:
# 
# ```python
# from onnxruntime.quantization import quantize_dynamic, QuantType
# 
# # Input & Output Paths
# input_model = "/home/user/YOLO_Assignment/Task_1/Model/runs_train_val/detect/yolov8_parking/weights/best.onnx"
# quantized_model = "best_quantized.onnx"
# 
# # Apply Dynamic Quantization
# quantize_dynamic(
#     model_input=input_model,
#     model_output=quantized_model,
#     weight_type=QuantType.QInt8  # Or use QuantType.QUInt8
# )
# 
# print("✅ Quantization complete. Saved as:", quantized_model)
# 

# In[5]:


from onnxruntime.quantization import quantize_dynamic, QuantType

# Input & Output Paths
input_model = "/home/user/YOLO_Assignment/Task_1/Model/runs_train_val/detect/yolov8_parking/weights/best.onnx"
quantized_model = "best_quantized.onnx"

# Apply Dynamic Quantization
quantize_dynamic(
    model_input=input_model,
    model_output=quantized_model,
    weight_type=QuantType.QInt8  # Or use QuantType.QUInt8
)

print(" Quantization complete. Saved as:", quantized_model)


# ### 📦 YOLOv8 Model Weights (Google Drive)
# 
# You can download the trained YOLOv8 model weights (`best.pt`, `best.onnx` and  `best_quantized.onnx`) from the following shared Google Drive folder:
# 
# 🔗 [Download YOLOv8 Weights](https://drive.google.com/drive/folders/1LfwMxtaqUxWGQxqFh6PKw3O1rZN8ohAV?usp=sharing)
# 
# ---
# 
# **Contents of the Folder:**
# - `best.pt`: Original PyTorch model (for Ultralytics API usage and re-exporting)
# - `best.onnx`: Exported ONNX model with postprocessing (NMS) enabled
# - `best_quantized.onnx`: Dynamically quantized ONNX model for efficient deployment
# 
# Please make sure to **replace `your_folder_id_here`** with the actual Google Drive folder ID.
# 

# In[ ]:




