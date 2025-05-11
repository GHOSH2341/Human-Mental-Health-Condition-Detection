# 🧠 Human Mental Health Condition Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

A real-time facial emotion detection app using deep learning. This project can be a prototype for detecting mental health conditions based on facial expressions.

---

## 📸 Features

- ✅ Upload an image and detect emotional expressions
- ✅ Recognize seven key emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- ✅ Save output images with bounding boxes and labels
- ✅ Clean UI using modern Streamlit design
- ✅ Uses Haar cascade for fast face detection
- ✅ Lightweight and responsive

---

## 🚀 Demo

![App Screenshot](https://github.com/GHOSH2341/Human-Mental-Health-Condition-Detection/blob/main/Screenshot%202025-05-11%20213712.png)
🌐 Try Now: [https://share.streamlit.io/](https://human-mental-health-condition-detection.streamlit.app/)
📦 Download the dataset used: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 💻 Tech Stack

| Technology     | Use Case                        |
|----------------|----------------------------------|
| Python         | Core programming language        |
| Streamlit      | UI rendering and deployment      |
| OpenCV         | Face detection                   |
| Keras/TensorFlow | Emotion classification model   |
| Pillow         | Image processing and saving      |

---

## 🗂️ Folder Structure

```bash
├── model.h5 # Trained deep learning model
├── haarcascade_frontalface_default.xml# Haar cascade classifier
├── emotion.py # Main Streamlit app
├── requirements.txt # List of dependencies
└── README.md # This file
```
---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/human-mental-health-detection.git](https://github.com/GHOSH2341/Human-Mental-Health-Condition-Detection.git)
cd human-mental-health-detection
```
### 2. Create Virtual Environment 
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt

```
### 4. Add Model & Cascade Files
- Place model.h5 (trained Keras model) in the root directory.

- Place haarcascade_frontalface_default.xml in the root directory.
### 5. Run the App
```bash
streamlit run emotion.py

```
# 🌍 Deployment Options
- 🟢 Streamlit Cloud
- Push code to a public GitHub repository
- Log in to Streamlit Cloud
- Deploy the app from your repository
- Add model and XML files as static assets or use environment variables/secrets

# 🟡 Hugging Face Spaces (Optional)
- Create a new Gradio or Streamlit Space
- Upload files and paste your emotion.py

## 🪪 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.





