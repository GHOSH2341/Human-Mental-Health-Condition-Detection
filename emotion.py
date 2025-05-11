import cv2
import streamlit as st
import numpy as np

# Set page config for a modern centered layout
st.set_page_config(page_title="Real-Time Face Detection", layout="centered")

st.title("🎥 Real-Time Face Detection with Streamlit")

# Sidebar controls
st.sidebar.header("Settings")
run = st.sidebar.checkbox("Start Webcam", value=False)
frame_window = st.image([])  # Placeholder for video frames

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize video capture object
cap = None

if run:
    if cap is None:
        cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw bounding boxes with a modern style
        for (x, y, w, h) in faces:
            # Rounded rectangle effect by drawing multiple rectangles with increasing thickness
            color = (0, 255, 0)  # Green box
            thickness = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            # Optional: Add label above face box
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert BGR to RGB for displaying in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels="RGB")
    else:
        st.error("Failed to capture video frame")
else:
    if cap is not None:
        cap.release()
        cap = None
    st.write("Click 'Start Webcam' to begin face detection")

# Cleanup on app exit
def cleanup():
    if cap is not None:
        cap.release()

st.on_event("shutdown", cleanup)
