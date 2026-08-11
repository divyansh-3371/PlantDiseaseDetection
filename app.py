import streamlit as st
import tensorflow as tf
import numpy as np
import os
import csv
from datetime import datetime
from PIL import Image

IMAGE_SIZE = 256
MODEL_PATH = "plant_disease_final_model.keras"  
DATASET_PATH = "E:/SEM5PROJECTS/oss/PlantVillage"
LOG_FILE = "prediction_log.csv"
CONFIDENCE_THRESHOLD = 70.0  

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial infection causing small, dark, water-soaked spots on leaves and fruit.",
        "remedy": "Remove infected plants, avoid overhead watering, apply copper-based bactericide, use disease-free seeds."
    },
    "Pepper__bell___healthy": {
        "description": "No visible signs of disease. Leaf appears healthy.",
        "remedy": "No treatment needed. Continue regular care and monitoring."
    },
    "Potato___Early_blight": {
        "description": "Fungal disease causing dark concentric-ring spots, usually on older leaves first.",
        "remedy": "Remove affected leaves, rotate crops yearly, apply fungicide (e.g. chlorothalonil), avoid wetting foliage."
    },
    "Potato___Late_blight": {
        "description": "Aggressive fungal-like infection causing dark, water-soaked lesions; can destroy crops rapidly in humid weather.",
        "remedy": "Destroy infected plants immediately, apply fungicide preventively, ensure good field drainage and airflow."
    },
    "Potato___healthy": {
        "description": "No visible signs of disease. Leaf appears healthy.",
        "remedy": "No treatment needed. Continue regular care and monitoring."
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial infection causing small, dark, greasy-looking spots on leaves and fruit.",
        "remedy": "Use copper-based sprays, avoid overhead irrigation, remove and destroy infected debris."
    },
    "Tomato_Early_blight": {
        "description": "Fungal infection causing brown spots with concentric rings, often starting on lower leaves.",
        "remedy": "Remove affected foliage, apply fungicide, mulch soil to reduce spore splash, rotate crops."
    },
    "Tomato_Late_blight": {
        "description": "Fast-spreading fungal-like disease causing large, irregular greasy patches, especially in cool, wet conditions.",
        "remedy": "Remove and destroy infected plants immediately, apply fungicide, avoid overhead watering."
    },
    "Tomato_Leaf_Mold": {
        "description": "Fungal disease common in humid greenhouse conditions, causing yellow patches on top of leaves and mold underneath.",
        "remedy": "Improve ventilation, reduce humidity, remove infected leaves, apply fungicide if severe."
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Fungal disease causing small circular spots with dark borders and grey centers.",
        "remedy": "Remove infected leaves, avoid overhead watering, apply fungicide, rotate crops."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Pest damage (not disease) from tiny mites causing stippled, yellowing leaves and fine webbing.",
        "remedy": "Spray with water to dislodge mites, use insecticidal soap or neem oil, introduce natural predators like ladybugs."
    },
    "Tomato__Target_Spot": {
        "description": "Fungal infection causing brown lesions with concentric rings, resembling a target pattern.",
        "remedy": "Remove infected leaves, improve air circulation, apply fungicide, avoid overhead watering."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Viral infection spread by whiteflies, causing upward leaf curling and yellowing.",
        "remedy": "Control whitefly population, remove infected plants, use resistant varieties, use reflective mulches."
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Viral infection causing mottled light/dark green patterns and leaf distortion.",
        "remedy": "Remove and destroy infected plants, disinfect tools, avoid tobacco contact, use resistant varieties."
    },
    "Tomato_healthy": {
        "description": "No visible signs of disease. Leaf appears healthy.",
        "remedy": "No treatment needed. Continue regular care and monitoring."
    },
}

DEFAULT_INFO = {
    "description": "No detailed information available for this class yet.",
    "remedy": "Consult a local agricultural expert for guidance."
}

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #555555;
        margin-top: 0px;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background-color: #f1f8e9;
        border-left: 6px solid #2e7d32;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        color: #1b1b1b !important;
    }
    .result-box b {
        color: #1b5e20 !important;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 6px solid #f9a825;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        color: #1b1b1b !important;
    }
    .warning-box b {
        color: #8a6d00 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### 🌿 Project Info")
    st.write("**AI-Based Plant Disease Detector**")
    st.write("B.Tech Project — JIIT Noida")
    st.write("Built with TensorFlow + Streamlit")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    input_mode = st.radio("Input method", ["Upload image(s)", "Use camera"])
    show_chart = st.checkbox("Show full probability chart", value=True)
    st.markdown("---")
    st.markdown("### 📊 About the model")
    st.write(f"- Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    st.write(f"- Confidence threshold: {CONFIDENCE_THRESHOLD}%")
    st.write("- Architecture: CNN (Conv2D + MaxPooling)")

st.markdown('<p class="main-header">🌿 Plant Disease Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload or capture leaf photos to detect disease, view treatment suggestions, '
    'and see model confidence.</p>',
    unsafe_allow_html=True
)

@st.cache_resource
def load_class_names():
    return sorted(os.listdir(DATASET_PATH))

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model_loaded = True
class_names_loaded = True

try:
    class_names = load_class_names()
except Exception as e:
    class_names_loaded = False
    class_names = []
    st.error(f"Could not read class names from '{DATASET_PATH}'.")
    st.exception(e)

try:
    model = load_model()
except Exception as e:
    model_loaded = False
    model = None
    st.error(f"Could not load model from '{MODEL_PATH}'. Make sure the file exists in this folder.")
    st.exception(e)

def predict(model, image: Image.Image):
    image = image.convert("RGB")
    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.utils.img_to_array(image_resized)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence, predictions[0]

def log_prediction(filename, predicted_class, confidence):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "filename", "predicted_class", "confidence"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), filename, predicted_class, confidence])

def render_result(image, filename, predicted_class, confidence, all_scores):
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.image(image, caption=filename, width="stretch")

    with col2:
        if confidence < CONFIDENCE_THRESHOLD:
            st.markdown(
                f"""<div class="warning-box">
                ⚠️ <b>Low confidence prediction ({confidence}%)</b><br>
                The model is not very sure about this result. Consider retaking the photo
                with better lighting, a closer/clearer view of the leaf, or consult an expert.
                </div>""",
                unsafe_allow_html=True
            )

        info = DISEASE_INFO.get(predicted_class, DEFAULT_INFO)

        st.markdown(
            f"""<div class="result-box">
            <b>Prediction:</b> {predicted_class.replace('_', ' ')}<br>
            <b>Confidence:</b> {confidence}%<br><br>
            <b>Description:</b> {info['description']}<br><br>
            <b>Suggested remedy:</b> {info['remedy']}
            </div>""",
            unsafe_allow_html=True
        )

        st.caption("⚠️ This prediction is for guidance only. Consult an agricultural expert for critical decisions.")

        with st.expander("Top 3 predictions"):
            top_indices = np.argsort(all_scores)[::-1][:3]
            for idx in top_indices:
                st.write(f"- {class_names[idx].replace('_', ' ')}: {round(100 * all_scores[idx], 2)}%")

        if show_chart:
            with st.expander("Full probability distribution"):
                chart_data = {class_names[i].replace('_', ' '): float(all_scores[i]) for i in range(len(class_names))}
                st.bar_chart(chart_data)

    log_prediction(filename, predicted_class, confidence)
    st.markdown("---")

if not (model_loaded and class_names_loaded):
    st.warning("Model or class names not loaded. Fix the errors above before continuing.")
else:
    if input_mode == "Upload image(s)":
        uploaded_files = st.file_uploader(
            "Choose one or more leaf images...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.write(f"**{len(uploaded_files)} image(s) uploaded.** Processing...")
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    predicted_class, confidence, all_scores = predict(model, image)
                render_result(image, uploaded_file.name, predicted_class, confidence, all_scores)
        else:
            st.write("👆 Upload one or more images to get started.")

    else:  
        camera_image = st.camera_input("Take a photo of a leaf")
        if camera_image is not None:
            image = Image.open(camera_image)
            with st.spinner("Analyzing captured photo..."):
                predicted_class, confidence, all_scores = predict(model, image)
            render_result(image, "camera_capture.jpg", predicted_class, confidence, all_scores)
        else:
            st.write("👆 Use the camera above to capture a leaf photo.")

if os.path.isfile(LOG_FILE):
    with st.expander("📄 View prediction log (all past predictions this app has made)"):
        with open(LOG_FILE, "r") as f:
            st.text(f.read())

st.markdown("---")
st.caption("Built with TensorFlow + Streamlit | Plant Disease Detection Model | JIIT Noida B.Tech Project")