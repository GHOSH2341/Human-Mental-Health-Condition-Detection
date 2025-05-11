import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import os
import time
import base64
from datetime import datetime


st.set_page_config(
    page_title="Mental Emotion Detection",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded"
)


primary_color = "#4F46E5"  # Indigo
secondary_color = "#7C3AED"  # Purple
light_bg = "#F9FAFB"
dark_bg = "#1E293B"
success_color = "#10B981"  # Green
warning_color = "#F59E0B"  # Amber
error_color = "#EF4444"  # Red


st.markdown("""
<style>
    /* Main elements */
    .main {
        background-color: #F9FAFB;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1E293B;
        font-family: 'Inter', sans-serif;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #4338CA;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transform: translateY(-1px);
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #4F46E5;
        border-radius: 12px;
        padding: 20px;
        background-color: rgba(79, 70, 229, 0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #F3F4F6;
    }
    
    /* Cards */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #4F46E5;
    }
    
    /* Status messages */
    .status-success {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10B981;
    }
    .status-warning {
        background-color: #FFFBEB;
        color: #92400E;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F59E0B;
    }
    .status-error {
        background-color: #FEF2F2;
        color: #B91C1C;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #EF4444;
    }
    
    /* Emotion labels */
    .emotion-label {
        color: white;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        display: inline-block;
        font-size: 0.875rem;
    }
    .emotion-angry {
        background-color: #EF4444;
    }
    .emotion-disgust {
        background-color: #84CC16;
    }
    .emotion-fear {
        background-color: #6366F1;
    }
    .emotion-happy {
        background-color: #F59E0B;
    }
    .emotion-sad {
        background-color: #64748B;
    }
    .emotion-surprise {
        background-color: #EC4899;
    }
    .emotion-neutral {
        background-color: #6B7280;
    }
    
    /* Statistics area */
    .stats-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
    }
    .stat-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        flex: 1;
        min-width: 120px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        text-align: center;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4F46E5;
    }
    
    /* Image display */
    .stImage {
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detection_resources():
    try:
        model = load_model('model.h5')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return model, face_cascade
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

model, face_cascade = load_detection_resources()


if model is None or face_cascade is None or face_cascade.empty():
    st.error("Critical error: Required detection resources couldn't be loaded. Please check file paths.")
    st.stop()


emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': '#EF4444',
    'Disgust': '#84CC16',
    'Fear': '#6366F1',
    'Happy': '#F59E0B',
    'Sad': '#64748B',
    'Surprise': '#EC4899',
    'Neutral': '#6B7280'
}


col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("Human Mental Health Condition Detection")
    st.markdown("Upload an image to detect mental health condition of faces")


with st.sidebar:
    st.header("Settings & Options")
    
   
    app_mode = st.radio(
        "Choose Mode",
        ["Single Image Analysis", "Batch Processing"]
    )
    
    st.markdown("---")
    
   s
    st.subheader("Detection Settings")
    
    detection_confidence = st.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for emotion detection"
    )
    
    scale_factor = st.slider(
        "Scale Factor",
        min_value=1.05,
        max_value=1.5,
        value=1.1,
        step=0.05,
        help="Scale factor for face detection"
    )
    
    min_neighbors = st.slider(
        "Minimum Neighbors",
        min_value=1,
        max_value=10,
        value=5,
        help="Minimum neighbors for face detection"
    )
    
    st.markdown("---")
    
    # File options
    st.subheader("File Options")
    save_images = st.checkbox("Save Detected Faces", value=False)
    
    if save_images:
        save_directory = st.text_input(
            "Save Directory",
            value="detected_faces",
            help="Directory where detected faces will be saved"
        )

        os.makedirs(save_directory, exist_ok=True)
    
    st.markdown("---")
    
    # Display options
    st.subheader("Display Options")
    show_confidence = st.checkbox("Show Confidence Score", value=True)
    show_bounding_boxes = st.checkbox("Show Bounding Boxes", value=True)
    bounding_box_thickness = st.slider("Box Thickness", 1, 5, 2)
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "Made By Aritrik Ghosh "
        "B.Tech CSE 2021-25."
        "Swami Vivekananda University"
    )

if app_mode == "Single Image Analysis":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image containing faces"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    

    if uploaded_file is not None:
      
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Progress simulation
            progress_text.text("Loading image...")
            progress_bar.progress(10)
            time.sleep(0.1)
            
         
            try:
                # Load and convert image
                progress_text.text("Processing image...")
                progress_bar.progress(30)
                
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4:  # If RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                
                # Convert image to grayscale for face detection
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                
                progress_text.text("Detecting faces...")
                progress_bar.progress(50)
                
                # Detect faces in the image
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(30, 30)
                )
                
                progress_text.text("Analyzing emotions...")
                progress_bar.progress(70)
                
                # Create a copy of the image for drawing
                display_img = img_array.copy()
                
                # Store detected emotions for statistics
                detected_emotions = []
                
                # If faces are detected
                if len(faces) > 0:
                    for i, (x, y, w, h) in enumerate(faces):
                        # Extract face ROI
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = img_array[y:y+h, x:x+w]
                        
                        # Preprocess for model
                        resized = cv2.resize(roi_gray, (48, 48))
                        normalized = resized / 255.0
                        reshaped = np.reshape(normalized, (1, 48, 48, 1))
                        
                        # Predict emotion
                        result = model.predict(reshaped, verbose=0)
                        emotion_idx = np.argmax(result)
                        emotion_label = emotions[emotion_idx]
                        confidence = float(result[0][emotion_idx])
                        
                        detected_emotions.append(emotion_label)
                        
                        # Draw bounding box if enabled
                        if show_bounding_boxes:
                            # Get color for the detected emotion
                            emotion_color = emotion_colors.get(emotion_label, (0, 255, 0))
                            # Convert hex to BGR
                            hex_color = emotion_color.lstrip('#')
                            rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # Reverse for BGR
                            
                            # Draw rectangle
                            cv2.rectangle(display_img, (x, y), (x+w, y+h), bgr_color, bounding_box_thickness)
                            
                            # Prepare label text
                            label_text = emotion_label
                            if show_confidence:
                                label_text += f" ({confidence:.2f})"
                            
                            # Calculate text background position and size
                            (text_width, text_height), _ = cv2.getTextSize(
                                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            
                            # Draw text background
                            cv2.rectangle(
                                display_img,
                                (x, y - text_height - 10),
                                (x + text_width, y),
                                bgr_color,
                                -1
                            )
                            
                            # Draw text
                            cv2.putText(
                                display_img,
                                label_text,
                                (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2
                            )
                        
                        # Save face image if selected
                        if save_images:
                            face_img = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(save_directory, f"{emotion_label}_{timestamp}_{i}.jpg")
                            face_img.save(filename)
                
                progress_text.text("Completed!")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_placeholder.empty()
                
                # Display results
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Results")
                    
                    if len(faces) == 0:
                        st.markdown(
                            '<div class="status-warning">'
                            '⚠️ No faces detected in the image. Try adjusting the detection settings or uploading a clearer image.'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        # Display detected image
                        st.image(display_img, use_column_width=True)
                        
                        # Show statistics
                        st.markdown("### Statistics")
                        
                        # Calculate emotion distribution
                        emotion_counts = {emotion: detected_emotions.count(emotion) for emotion in emotions}
                        total_faces = len(faces)
                        
                        # Display emotion distribution in a nice format
                        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                        
                        for emotion in emotions:
                            count = emotion_counts.get(emotion, 0)
                            if count > 0:
                                percentage = (count / total_faces) * 100
                                st.markdown(
                                    f'<div class="stat-card">'
                                    f'<div class="stat-value">{count}</div>'
                                    f'<div class="emotion-label" style="background-color: {emotion_colors[emotion]}">{emotion}</div>'
                                    f'<div>{percentage:.1f}%</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display save confirmation if applicable
                        if save_images and len(faces) > 0:
                            st.markdown(
                                f'<div class="status-success">'
                                f'✅ Saved {len(faces)} detected face images to "{save_directory}" folder.'
                                f'</div>',
                                unsafe_allow_html=True
                            )
            
            except Exception as e:
                progress_placeholder.empty()
                st.markdown(
                    f'<div class="status-error">'
                    f'❌ Error processing image: {e}'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        # When no image is uploaded, show placeholder
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("Please upload an image to start the emotion detection process.")
            st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "Batch Processing":
    # Batch processing interface
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Batch Processing")
    st.markdown("Upload multiple images for batch emotion analysis.")
    
    uploaded_files = st.file_uploader(
        "Choose images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload multiple clear images containing faces"
    )
    
    if uploaded_files:
        if st.button("Process All Images"):
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each uploaded file
            results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                progress = (i / len(uploaded_files))
                progress_bar.progress(progress)
                status_text.text(f"Processing image {i+1} of {len(uploaded_files)}: {uploaded_file.name}")
                
                try:
                    # Open and process the image
                    image = Image.open(uploaded_file)
                    img_array = np.array(image)
                    
                    # Check if image is grayscale and convert accordingly
                    if len(img_array.shape) == 2:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                    elif img_array.shape[2] == 4:  # If RGBA
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                    
                    # Convert image to grayscale for face detection
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces in the image
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=(30, 30)
                    )
                    
                    # Create a copy of the image for drawing
                    display_img = img_array.copy()
                    
                    # Store detected emotions for this image
                    image_emotions = []
                    
                    # If faces are detected
                    if len(faces) > 0:
                        for j, (x, y, w, h) in enumerate(faces):
                            # Extract face ROI
                            roi_gray = gray[y:y+h, x:x+w]
                            roi_color = img_array[y:y+h, x:x+w]
                            
                            # Preprocess for model
                            resized = cv2.resize(roi_gray, (48, 48))
                            normalized = resized / 255.0
                            reshaped = np.reshape(normalized, (1, 48, 48, 1))
                            
                            # Predict emotion
                            result = model.predict(reshaped, verbose=0)
                            emotion_idx = np.argmax(result)
                            emotion_label = emotions[emotion_idx]
                            confidence = float(result[0][emotion_idx])
                            
                            image_emotions.append(emotion_label)
                            
                            # Draw bounding box if enabled
                            if show_bounding_boxes:
                                # Get color for the detected emotion
                                emotion_color = emotion_colors.get(emotion_label, (0, 255, 0))
                                # Convert hex to BGR
                                hex_color = emotion_color.lstrip('#')
                                rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                                bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # Reverse for BGR
                                
                                # Draw rectangle
                                cv2.rectangle(display_img, (x, y), (x+w, y+h), bgr_color, bounding_box_thickness)
                                
                                # Draw label
                                label_text = emotion_label
                                if show_confidence:
                                    label_text += f" ({confidence:.2f})"
                                
                                # Calculate text background position and size
                                (text_width, text_height), _ = cv2.getTextSize(
                                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                                )
                                
                                # Draw text background
                                cv2.rectangle(
                                    display_img,
                                    (x, y - text_height - 10),
                                    (x + text_width, y),
                                    bgr_color,
                                    -1
                                )
                                
                                # Draw text
                                cv2.putText(
                                    display_img,
                                    label_text,
                                    (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 255),
                                    2
                                )
                            
                            # Save face image if selected
                            if save_images:
                                face_img = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = os.path.join(save_directory, f"{uploaded_file.name}_{emotion_label}_{timestamp}_{j}.jpg")
                                face_img.save(filename)
                    
                    # Convert the image to RGB for display
                    display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                    
                    # Append results for this image
                    results.append({
                        "filename": uploaded_file.name,
                        "faces_count": len(faces),
                        "emotions": image_emotions,
                        "processed_image": display_img_rgb
                    })
                
                except Exception as e:
                    # Append error result
                    results.append({
                        "filename": uploaded_file.name,
                        "error": str(e)
                    })
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            # Display batch results
            st.markdown("## Batch Results")
            
            # Summary stats
            total_images = len(results)
            total_faces = sum([result.get("faces_count", 0) for result in results])
            failed_images = sum([1 for result in results if "error" in result])
            
            # Display summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Images", total_images)
            
            with col2:
                st.metric("Total Faces", total_faces)
            
            with col3:
                st.metric("Failed Images", failed_images)
            
            # Emotion distribution in all images
            all_emotions = []
            for result in results:
                if "emotions" in result:
                    all_emotions.extend(result["emotions"])
            
            if all_emotions:
                st.markdown("### Overall Emotion Distribution")
                
                emotion_counts = {emotion: all_emotions.count(emotion) for emotion in emotions}
                
                # Display emotion distribution
                st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                
                for emotion in emotions:
                    count = emotion_counts.get(emotion, 0)
                    if count > 0:
                        percentage = (count / len(all_emotions)) * 100
                        st.markdown(
                            f'<div class="stat-card">'
                            f'<div class="stat-value">{count}</div>'
                            f'<div class="emotion-label" style="background-color: {emotion_colors[emotion]}">{emotion}</div>'
                            f'<div>{percentage:.1f}%</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display individual results
            st.markdown("### Individual Results")
            
            for i, result in enumerate(results):
                with st.expander(f"Image {i+1}: {result['filename']}"):
                    if "error" in result:
                        st.error(f"Processing error: {result['error']}")
                    else:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(result["processed_image"], caption=result["filename"])
                        
                        with col2:
                            st.write(f"Faces detected: {result['faces_count']}")
                            
                            if result["faces_count"] > 0:
                                # Count emotions in this image
                                image_emotion_counts = {emotion: result["emotions"].count(emotion) for emotion in emotions}
                                
                                # Display emotions detected
                                st.write("Emotions detected:")
                                for emotion, count in image_emotion_counts.items():
                                    if count > 0:
                                        st.markdown(
                                            f'<span class="emotion-label" style="background-color: {emotion_colors[emotion]}">'
                                            f'{emotion}: {count}</span> ',
                                            unsafe_allow_html=True
                                        )
            
            # Save notification if applicable
            if save_images and total_faces > 0:
                st.markdown(
                    f'<div class="status-success">'
                    f'✅ Saved {total_faces} detected face images to "{save_directory}" folder.'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("Upload multiple images to begin batch processing.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
footer_html = """
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #6B7280;">
    <p>Made with ❤️ by Aritrik Ghosh</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
