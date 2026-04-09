import streamlit as st
import numpy as np
import json
from keras.models import load_model
from tensorflow.keras.preprocessing import image

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    layout="centered"
)

# ------------------ PREMIUM UI ------------------
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #1b5e20;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 25px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(to right, #66bb6a, #43a047);
        color: white;
        text-align: center;
        font-size: 18px;
    }
    .footer {
        text-align: center;
        color: gray;
        font-size: 14px;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown('<div class="title">🌿 Plant Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered leaf disease detection system</div>', unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("🌿 About")
st.sidebar.info("This app uses a deep learning model (MobileNetV2) to detect plant diseases from leaf images.")

# ------------------ LOAD CLASS NAMES ------------------
with open("class_names.json") as f:
    class_names = json.load(f)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model_app():
    return load_model("plant_disease_model.h5")

model = load_model_app()

# ------------------ MAIN CARD ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # ------------------ PREPROCESS ------------------
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ------------------ PREDICTION ------------------
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # Clean name
    clean_name = predicted_class.replace("___", " - ").replace("_", " ")

    # ------------------ RESULT ------------------
    with col2:
        st.markdown(f"""
        <div class="result-box">
            <h3>🧠 Prediction</h3>
            <p><b>{clean_name}</b></p>
            <p>Confidence: {confidence:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # ------------------ CONFIDENCE BAR ------------------
    st.markdown("### 📊 Confidence Level")
    st.progress(int(confidence * 100))

    # ------------------ REMEDIES ------------------
    st.markdown("### 🌿 Suggested Remedy")

    remedies = {
        "Early blight": "Remove infected leaves and apply fungicide.",
        "Late blight": "Avoid overwatering and use copper-based fungicide.",
        "Bacterial spot": "Use disease-free seeds and avoid overhead watering.",
        "Leaf Mold": "Ensure proper air circulation and reduce humidity.",
        "Spider mites": "Spray neem oil or insecticidal soap.",
        "Target Spot": "Remove affected leaves and apply fungicide.",
        "YellowLeaf Curl": "Control whiteflies and remove infected plants.",
        "Mosaic virus": "Remove infected plants and sanitize tools.",
        "healthy": "Your plant is healthy! 🌱 Keep it up!"
    }

    found = False
    for key in remedies:
        if key.lower() in clean_name.lower():
            st.success(remedies[key])
            found = True
            break

    if not found:
        st.info("Maintain proper watering, sunlight, and plant hygiene.")

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown('<div class="footer">Made with ❤️ using Deep Learning & Streamlit</div>', unsafe_allow_html=True)
