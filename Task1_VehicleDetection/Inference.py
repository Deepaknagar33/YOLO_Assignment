#!/usr/bin/env python
# coding: utf-8

# ## 🚗 YOLOv8 Inference & Evaluation on Parking Slot Detection
# 
# This script performs inference and evaluation using a trained YOLOv8 model on a test dataset for detecting **empty** and **occupied** parking slots. Below is a breakdown of the script:
# 
# ---
# 
# ### 🔧 Configuration
# - **Model Path:** `/home/user/YOLO_Assignment/Task_1/Model/runs_train_val/detect/yolov8_parking/weights/best.pt`
# - **Data YAML:** `/home/user/YOLO_Assignment/Task_1/Dataset/dataset_yolo/data.yaml`
# - **Test Image Directory:** `/home/user/YOLO_Assignment/Task_1/Dataset/dataset_yolo/test/images`
# - **Results File:** `test_results_final.txt`
# - **Output Directory for Results:** `test_outputs`
# - **Device Used:** Automatically selects **GPU** (`cuda:0`) if available, else **CPU**
# 
# ---
# 
# ### 📦 Step-by-Step Process
# 
# 1. **Load YOLOv8 Model**  
#    Using the trained `best.pt` model from a previous run.
# 
# 2. **Create Output Folder**  
#    Ensures the test output directory exists for saving visual predictions.
# 
# 3. **Inference on Test Images**  
#    - Loads all `.jpg` images from the test directory
#    - Runs YOLOv8 inference with:
#      - Image size: 640
#      - Confidence threshold: 0.25
#      - Output images saved to the `test_outputs` directory
# 
# 4. **Calculate FPS**  
#    Measures the average inference speed over all test images.
# 
# 5. **GPU Memory Used**  
#    Reports how much GPU memory was used during inference (if CUDA is available).
# 
# 6. **Model Info and Metrics**
#    - Calls `model.info()` to get architecture details
#    - Evaluates model on test set using `model.val()`:
#      - Computes overall precision, recall, mAP@0.5, mAP@0.5:0.95
#      - Also extracts per-class metrics for:
#        - `space-empty` (empty parking slot)
#        - `space-occupied` (occupied parking slot)
# 
# 7. **Log Results to File**  
#    Writes all metrics and info to `test_results_final.txt`.
# 
# ---
# 
# ### 📈 Output Summary
# 
# The results saved to `test_results_final.txt` include:
# - Mean precision, recall, mAP scores
# - Per-class precision and recall
# - Model structure info
# - Inference speed (FPS)
# - GPU memory used
# 
# ---
# 
# ### ✅ Output
# Results printed to console and saved to:
# 

# In[3]:


from ultralytics import YOLO
import torch
import time
from glob import glob
import os

# === CONFIGURATION ===
MODEL_PATH = '/home/user/YOLO_Assignment/Task_1/Model/runs_train_val/detect/yolov8_parking/weights/best.pt'
DATA_YAML = '/home/user/YOLO_Assignment/Task_1/Dataset/dataset_yolo/data.yaml'
TEST_DIR = '/home/user/YOLO_Assignment/Task_1/Dataset/dataset_yolo/test/images'
LOG_FILE = 'test_results_final.txt'
SAVE_DIR = 'test_outputs'
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === MAKE OUTPUT FOLDER ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === INFERENCE ON TEST IMAGES ===
test_images = sorted(glob(f'{TEST_DIR}/*.jpg'))
start = time.time()
for img_path in test_images:
    model.predict(
        source=img_path,
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=False,
        save_crop=False,
        name=SAVE_DIR,
        device=DEVICE,
        verbose=False
    )
end = time.time()

# === FPS ===
fps = len(test_images) / (end - start)

# === GPU MEMORY ===
gpu_mem = torch.cuda.memory_allocated(DEVICE) / (1024 ** 3) if torch.cuda.is_available() else 0

# === MODEL INFO ===
model_info = model.info(verbose=True)

# === PRECISION, RECALL, mAP ===
metrics = model.val(data=DATA_YAML, device=DEVICE, imgsz=640, save=False, verbose=False)

# Overall metrics
precision = sum(metrics.box.p) / len(metrics.box.p)
recall = sum(metrics.box.r) / len(metrics.box.r)
map50 = sum(metrics.box.ap50) / len(metrics.box.ap50)
map = sum(metrics.box.ap) / len(metrics.box.ap)

# === PER-CLASS METRICS ===
class_names = metrics.names  # dictionary: {0: 'space-empty', 1: 'space-occupied'}
per_class_precision = metrics.box.p
per_class_recall = metrics.box.r

try:
    class_occupied = [k for k, v in class_names.items() if v == 'space-occupied'][0]
    class_empty = [k for k, v in class_names.items() if v == 'space-empty'][0]

    precision_occupied = per_class_precision[class_occupied].item()
    recall_occupied = per_class_recall[class_occupied].item()
    precision_empty = per_class_precision[class_empty].item()
    recall_empty = per_class_recall[class_empty].item()
except IndexError:
    precision_occupied = recall_occupied = precision_empty = recall_empty = -1.0
    print("❌ One or more class names not found in metrics.names!")

# === WRITE TO LOG FILE ===
with open(LOG_FILE, 'w') as f:
    f.write("=== YOLOv8 Test Summary ===\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
    f.write(f"Inference FPS: {fps:.2f}\n")
    f.write(f"GPU Memory Used: {gpu_mem:.2f} GB\n")
    f.write(f"Mean Precision: {precision:.4f}\n")
    f.write(f"Mean Recall: {recall:.4f}\n")
    f.write(f"mAP@0.5: {map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {map:.4f}\n")
    f.write("\n=== Per-Class Precision & Recall ===\n")
    f.write(f"Parked Vehicles (space-occupied) - Precision: {precision_occupied:.4f}, Recall: {recall_occupied:.4f}\n")
    f.write(f"Empty Slots (space-empty)       - Precision: {precision_empty:.4f}, Recall: {recall_empty:.4f}\n")
    f.write("\nModel Info:\n")
    f.write(str(model_info))
    f.write("\n=============================\n")

print("✅ Inference & evaluation complete. Results saved to:", LOG_FILE)


# # YOLOv8 Parking Space Detection - Test Results
# 
# ## Model Performance Summary
# 
# - **Model Path**: `/home/user/YOLO_Assignment/Task_1/Model/runs_train_val/detect/yolov8_parking/weights/best.pt`
# - **Device**: CUDA
# - **Inference Speed**: 17.02 FPS
# - **GPU Memory Usage**: 0.02 GB
# - **Mean Precision**: 0.9985
# - **Mean Recall**: 0.9956
# - **mAP@0.5**: 0.9946
# - **mAP@0.5:0.95**: 0.9823
# 
# ## Detailed Metrics
# 
# | Metric          | Value   |
# |-----------------|---------|
# | Images Processed | 7,272   |
# | Instances       | 429,948 |
# | Box Precision   | 0.999   |
# | Box Recall      | 0.996   |
# | mAP50           | 0.995   |
# | mAP50-95        | 0.982   |
# 
# ## Speed Analysis
# 
# | Phase          | Time per Image |
# |----------------|----------------|
# | Preprocess     | 0.1 ms         |
# | Inference      | 3.4 ms         |
# | Loss           | 0.0 ms         |
# | Postprocess    | 1.5 ms         |
# 
# ## Per-Class Performance
# 
# | Class                     | Precision | Recall  |
# |---------------------------|-----------|---------|
# | Parked Vehicles (occupied)| 0.9984    | 0.9945  |
# | Empty Slots               | 0.9986    | 0.9967  |
# 
# ## Model Information
# 
# - Architecture: (72, 3006038, 0, 8.086272)
# 
# > Results saved to: `runs/detect/val`

# ## YOLOv8 Test Set Detection Outputs
# 
# ## Output Visualizations
# 
# The detection outputs from the test set are available for review:
# 
# 📁 [View Test Set Detection Outputs on Google Drive](https://drive.google.com/drive/folders/1SE_JZDAUbTKTk64N4kbA6v3WweebN5Bo?usp=sharing)

# ## 🎥 YOLOv8 Inference on Video/Image for Parking Slot Detection
# 
# This section performs inference using a trained YOLOv8 model on either a **video file** (e.g., `.mp4`) or a **single image**. The model detects and classifies parking slots into two categories:  
# - `space-empty` (class ID 0)  
# - `space-occupied` (class ID 1)
# 
# ---
# 
# ### ⚙️ Configuration
# 
# - **Model Path:**  
#   `/home/user/YOLO_Assignment/Task_1/Model/runs/detect/yolov8_parking/weights/best.pt`
# 
# - **Input Video/Image:**  
#   `/home/user/YOLO_Assignment/Task_1/Dataset/sample_parking.mp4`
# 
# - **Device Used:**  
#   Automatically selects `CUDA` if available, otherwise falls back to `CPU`.
# 
# - **Output Files:**  
#   - Annotated frames and summary video saved to: `inference_outputs/`
#   - Inference log: `inference_outputs/inference_log.txt`
#   - Frame-wise slot counts: `inference_outputs/frame_counts.txt`
# 
# ---
# 
# ### 🚀 Inference Flow
# 
# 1. **Model Loading:** YOLOv8 model is loaded using Ultralytics API.
# 2. **Input Handling:**
#    - If the input is a **video**, it reads each frame and applies detection.
#    - If the input is a **single image**, it processes it directly.
# 3. **Detection & Annotation:**
#    - The model predicts bounding boxes per frame.
#    - Counts `space-empty` and `space-occupied` slots.
#    - Adds the counts as overlay text on the frame.
# 4. **Saving Outputs:**
#    - Annotated frames saved as `.jpg`.
#    - Frame-wise slot counts logged.
#    - A summary video with annotations is generated (if input is a video).
# 5. **Performance Metrics:**
#    - Calculates average FPS and GPU memory used.
# 
# ---
# 
# 
# 
# 

# In[2]:


from ultralytics import YOLO
import torch
import time
import cv2
import os

# === CONFIGURATION ===
MODEL_PATH = '/home/user/YOLO_Assignment/Task_1/Model/runs/detect/yolov8_parking/weights/best.pt'
DATA_YAML = '/home/user/YOLO_Assignment/Task_1/Dataset/dataset_yolo/data.yaml'
INPUT_PATH ='/home/user/YOLO_Assignment/Task_1/Dataset/sample_parking.mp4'
OUTPUT_DIR = 'inference_outputs'
LOG_FILE = os.path.join(OUTPUT_DIR, 'inference_log.txt')
FRAME_COUNTS_FILE = os.path.join(OUTPUT_DIR, 'frame_counts.txt')
SUMMARY_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'summary_output.mp4')
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# === CREATE OUTPUT DIR ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === HELPER: PROCESS FRAME ===
def process_frame(frame, frame_id=None, writer=None, count_writer=None):
    results = model.predict(source=frame, device=DEVICE, conf=0.25, save=False, verbose=False)[0]

    total_empty = 0
    total_occupied = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:
            total_empty += 1
        elif cls == 1:
            total_occupied += 1

    annotated = results.plot()
    overlay_text = f"Empty: {total_empty} | Occupied: {total_occupied}"
    cv2.putText(annotated, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if frame_id is not None:
        img_name = f"frame_{frame_id:04d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), annotated)
        if count_writer:
            count_writer.write(f"{img_name} --> Empty: {total_empty}, Occupied: {total_occupied}\n")

    if writer:
        writer.write(annotated)

    return total_empty, total_occupied

# === INFERENCE ===
is_video = INPUT_PATH.lower().endswith(('.mp4', '.avi', '.mov'))
start = time.time()

avg_empty = avg_occupied = 0
frame_count = 0

with open(FRAME_COUNTS_FILE, 'w') as count_writer:
    if is_video:
        cap = cv2.VideoCapture(INPUT_PATH)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(SUMMARY_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps_input,
                                 (frame_width, frame_height))

        total_empty_sum = 0
        total_occupied_sum = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            empty, occupied = process_frame(frame, frame_id=frame_count, writer=writer, count_writer=count_writer)
            total_empty_sum += empty
            total_occupied_sum += occupied
            frame_count += 1

        cap.release()
        writer.release()

        avg_empty = total_empty_sum // frame_count
        avg_occupied = total_occupied_sum // frame_count

    else:
        frame = cv2.imread(INPUT_PATH)
        frame_count = 1
        avg_empty, avg_occupied = process_frame(frame, frame_id=0, count_writer=count_writer)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'output.jpg'), frame)

end = time.time()

# === METRICS ===
fps = frame_count / (end - start)
gpu_mem = torch.cuda.memory_allocated(DEVICE) / (1024 ** 3) if torch.cuda.is_available() else 0

# === LOGGING ===
with open(LOG_FILE, 'w') as f:
    f.write("=== Inference Results ===\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
    f.write(f"Input: {INPUT_PATH}\n")
    f.write(f"FPS: {fps:.2f}\n")
    f.write(f"GPU Memory Used: {gpu_mem:.2f} GB\n")
    f.write(f"Total Occupied Slots: {avg_occupied}\n")
    f.write(f"Total Empty Slots: {avg_empty}\n")
    f.write("=========================\n")

# === COMPLETION MESSAGE ===
print(" Inference complete.")
print(f" Log saved at: {LOG_FILE}")
print(f" Frame-wise counts saved at: {FRAME_COUNTS_FILE}")
print(f" Output images saved at: {OUTPUT_DIR}")
if is_video:
    print(f" Summary video saved at: {SUMMARY_VIDEO_PATH}")


# ## 📂 Downloadable Resources for Inference Demo
# 
# Below are links to **sample inputs**, the **output video after detection**, and **detailed inference results**, including per-frame bounding box images and logs.
# 
# ---
# 
# ### 🔗 Sample Input Video
# 🎥 A test video demonstrating parking lot occupancy status.
# 
# 👉 [Click to Download Sample Input Video](https://drive.google.com/drive/folders/1SSkWE9dVIxG6V5gL_z4RCzgiRNpLKZWH?usp=sharing)
# 
# ---
# 
# ### 🔗 Output Summary Video (With Detections)
# 📽️ YOLOv8-inferred video showing bounding boxes and class labels for each frame.
# 
# 👉 [Click to Download Output Result Video](https://drive.google.com/drive/folders/1BgvYgd7-cGmmOBeb2eqhkq-3wTkT6URn?usp=sharing)
# 
# ---
# 
# ### 🔗 Annotated Frames, Logs & Count Files
# 📁 This archive contains:
# - Annotated frames with bounding boxes
# - `inference_log.txt` summarizing FPS, GPU usage, slot counts
# - `frame_counts.txt` listing per-frame empty/occupied slot data
# 
# 👉 [Click to Download Output Frames and Logs](https://drive.google.com/drive/folders/10KqXTMcfNwmrX5cxEsTpXIl2gxlMuG2w?usp=sharing)
# 
# ---
# 
# ### 📌 Notes
# - Each output frame includes overlay text like `Empty: 23 | Occupied: 17`.
# - Suitable for visual validation of detection quality and performance.
# 
# Make sure you're logged in to your Google account if access is restricted.
# 

# ## 📊 YOLOv8 Parking Detection: Inference Results on Sample Video
# 
# This section presents the output of a YOLOv8 model used to detect **occupied** and **empty** parking slots in a sample video.
# 
# ---
# 
# ### 🧠 Model & Environment Configuration
# 
# | Item              | Value                                                                 |
# |-------------------|-----------------------------------------------------------------------|
# | **Model Path**     | `/home/user/YOLO_Assignment/Task_1/Model/runs/detect/yolov8_parking/weights/best.pt` |
# | **Device Used**    | CUDA (GPU)                                                            |
# | **Input Source**   | `/home/user/YOLO_Assignment/Task_1/Dataset/sample_parking.mp4`       |
# | **Frames Processed** | 132+ frames                                                        |
# | **FPS**            | 13.06                                                                 |
# | **GPU Memory Used**| 0.02 GB                                                               |
# | **Avg Occupied Slots** | 4                                                                |
# | **Avg Empty Slots**    | 0                                                                |
# 
# ---
# 
# 
# 

# # Top 10 Frames with Highest Occupied Slots
# 
# | Frame Name     | Empty | Occupied |
# |----------------|--------|-----------|
# | frame_0131.jpg |   1    |    17     |
# | frame_0152.jpg |   0    |    14     |
# | frame_0151.jpg |   1    |    13     |
# | frame_0136.jpg |   1    |    13     |
# | frame_0159.jpg |   0    |    12     |
# | frame_0133.jpg |   1    |    12     |
# | frame_0134.jpg |   1    |    11     |
# | frame_0153.jpg |   0    |    12     |
# | frame_0135.jpg |   1    |    10     |
# | frame_0106.jpg |   0    |    10     |
# 

# In[ ]:




