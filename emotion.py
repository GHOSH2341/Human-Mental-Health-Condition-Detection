import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import os
import time

# Set Streamlit page config as the first command
st.set_page_config(page_title="Emotion Detection from Image", layout="centered")

# Load model and Haar Cascade
model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Custom CSS for modern UI
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f9;
            padding: 10px;
        }
        .stTitle {
            text-align: center;
            font-size: 36px;
            font-weight: 700;
            color: #333;
        }
        .stMarkdown {
            text-align: center;
            font-size: 20px;
            color: #555;
        }
        .stSidebar {
            background-color: #2C3E50;
            color: #fff;
        }
        .stButton, .stCheckbox {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton:hover, .stCheckbox:hover {
            background-color: #2980b9;
        }
        .stImage {
            border: 5px solid #3498db;
            border-radius: 10px;
            padding: 5px;
        }
        .stAlert {
            background-color: #e74c3c;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .stSuccess {
            background-color: #2ecc71;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .stWarning {
            background-color: #f39c12;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<p class="stTitle">🧠 Facial Emotion Detection from Uploaded Image</p>', unsafe_allow_html=True)
st.markdown('<p class="stMarkdown">Upload an image to detect the emotion of faces present in it.</p>', unsafe_allow_html=True)

# Sidebar options
st.sidebar.header("Options")
save_image = st.sidebar.checkbox("Save Detected Face Image")
image_dir = "captured_faces"
os.makedirs(image_dir, exist_ok=True)

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If faces are detected
    if len(faces) == 0:
        st.markdown('<div class="stAlert">No faces detected in the image.</div>', unsafe_allow_html=True)
    
    # Process each face in the image
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img_array[y:y+h, x:x+w]
        resized = cv2.resize(roi_gray, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        result = model.predict(reshaped, verbose=0)
        label = emotions[np.argmax(result)]

        # Draw bounding box and label on the face
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_array, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Save face image if selected
        if save_image:
            face_img = Image.fromarray(roi_color)
            filename = os.path.join(image_dir, f"{label}_{int(time.time())}.jpg")
            face_img.save(filename)
            st.markdown(f'<div class="stSuccess">Saved face image: {filename}</div>', unsafe_allow_html=True)

    # Display image with bounding box and label
    st.image(img_array, caption="Processed Image", use_container_width=True)

else:
    st.markdown('<div class="stWarning">Please upload an image to start the emotion detection.</div>', unsafe_allow_html=True)
