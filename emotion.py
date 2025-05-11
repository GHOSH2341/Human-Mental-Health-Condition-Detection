import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import os

# Load model and Haar Cascade
model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.set_page_config(page_title="Emotion Detection from Image", layout="centered")
st.title("🧠 Facial Emotion Detection from Uploaded Image")
st.markdown("Upload an image to detect the emotion of faces present in it.")

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
        st.warning("No faces detected in the image.")
    
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
            st.sidebar.success(f"Saved face image: {filename}")

    # Display image with bounding box and label
    st.image(img_array, caption="Processed Image", use_column_width=True)

else:
    st.info("Please upload an image to start the emotion detection.")
