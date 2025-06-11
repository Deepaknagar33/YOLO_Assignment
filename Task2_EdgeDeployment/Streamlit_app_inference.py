import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import tempfile
import time
import psutil
import GPUtil
from PIL import Image

# === CONFIG ===
ONNX_PATH = "best.onnx"
CLASS_NAMES = ['space-empty', 'space-occupied']  # index 0 = empty, 1 = occupied
CONF_THRESH = 0.25

# === Load ONNX Model ===
@st.cache_resource
def load_model():
    return ort.InferenceSession(ONNX_PATH)

session = load_model()

# === Streamlit Layout ===
st.title("🅿️ Parking Slot Detection - Image/Video (ONNX + Streamlit)")

input_mode = st.radio("Select Input Type", ['Image', 'Video'])

if input_mode == 'Image':
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_image.read())
        image_path = tfile.name

        orig_img = cv2.imread(image_path)
        h0, w0 = orig_img.shape[:2]
        img = cv2.resize(orig_img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

        # === Inference ===
        start = time.time()
        outputs = session.run(None, {"images": img.astype(np.float32)})
        end = time.time()

        fps = 1 / (end - start)
        cpu = psutil.cpu_percent()
        gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0

        # === Postprocessing ===
        detections = outputs[0][0]
        counts = {'space-empty': 0, 'space-occupied': 0}
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf < CONF_THRESH:
                continue
            x1 = int(x1 * w0 / 640)
            y1 = int(y1 * h0 / 640)
            x2 = int(x2 * w0 / 640)
            y2 = int(y2 * h0 / 640)
            class_id = int(cls)
            label = f"{CLASS_NAMES[class_id]}: {conf:.2f}"
            counts[CLASS_NAMES[class_id]] += 1
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        overlay = f"FPS: {fps:.2f} | CPU: {cpu}% | GPU: {gpu}MB"
        cv2.putText(orig_img, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        st.image(orig_img, caption="Detection Output", channels="BGR", use_column_width=True)
        st.info(f"✅ Empty Slots: {counts['space-empty']} | Occupied Slots: {counts['space-occupied']}")

elif input_mode == 'Video':
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h0, w0 = frame.shape[:2]
            img = cv2.resize(frame, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img /= 255.0
            img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

            # Inference
            start = time.time()
            outputs = session.run(None, {"images": img.astype(np.float32)})
            end = time.time()

            fps = 1 / (end - start)
            cpu = psutil.cpu_percent()
            gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0

            # Postprocess
            detections = outputs[0][0]
            counts = {'space-empty': 0, 'space-occupied': 0}
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf < CONF_THRESH:
                    continue
                x1 = int(x1 * w0 / 640)
                y1 = int(y1 * h0 / 640)
                x2 = int(x2 * w0 / 640)
                y2 = int(y2 * h0 / 640)
                class_id = int(cls)
                label = f"{CLASS_NAMES[class_id]}: {conf:.2f}"
                counts[CLASS_NAMES[class_id]] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            overlay = f"FPS: {fps:.2f} | CPU: {cpu}% | GPU: {gpu}MB"
            cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

            stframe.image(frame, channels="BGR", use_column_width=True)
        cap.release()
        st.success("✅ Video processed completely.")
