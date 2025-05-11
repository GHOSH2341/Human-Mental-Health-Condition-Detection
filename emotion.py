import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import time
import tempfile
import os

# Load model and Haar Cascade
model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.set_page_config(page_title="Real-time Emotion Detection", layout="centered")
st.title("🧠 Real-time Facial Emotion Detection")
st.markdown("Using your webcam to detect facial expressions in real-time.")

# Sidebar options
st.sidebar.header("Options")
capture_image = st.sidebar.button("📸 Capture Current Frame")
save_image = st.sidebar.checkbox("Save Detected Face Image")
image_dir = "captured_faces"
os.makedirs(image_dir, exist_ok=True)

# Start webcam
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Temporary display delay
time.sleep(1)

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            resized = cv2.resize(roi_gray, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))
            result = model.predict(reshaped, verbose=0)
            label = emotions[np.argmax(result)]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Save face image if selected
            if save_image:
                face_img = Image.fromarray(roi_color)
                filename = os.path.join(image_dir, f"{label}_{int(time.time())}.jpg")
                face_img.save(filename)

        # Convert BGR to RGB for Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if capture_image:
            filename = os.path.join(image_dir, f"capture_{int(time.time())}.jpg")
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(filename)
            st.sidebar.success(f"Captured image saved to: {filename}")
            break

except Exception as e:
    st.error(f"Error occurred: {str(e)}")
finally:
    camera.release()
