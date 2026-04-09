# 🌿 Plant Disease Detection System

An AI-powered web application that detects plant diseases from leaf images using **Deep Learning (CNN + MobileNetV2)** and provides real-time predictions with confidence scores.

---

## 🚀 Project Overview

This project leverages **transfer learning** with MobileNetV2 to classify plant leaf images into multiple disease categories.
A user-friendly **Streamlit web interface** allows users to upload images and instantly receive predictions along with suggested remedies.

---

## 🧠 Key Features

* 🌿 Multi-class plant disease classification
* ⚡ Real-time image prediction
* 📊 Confidence score visualization
* 🖼️ Image upload interface
* 💡 Suggested remedies for detected diseases
* 🎨 Clean and interactive UI using Streamlit

---

## 🏗️ Tech Stack

* **Deep Learning:** CNN, MobileNetV2 (Transfer Learning)
* **Frameworks:** TensorFlow / Keras
* **Frontend:** Streamlit
* **Libraries:** NumPy, Pillow, Matplotlib, Scikit-learn

---

## 📂 Dataset

* Used the **PlantVillage Dataset**
* Contains thousands of labeled images of healthy and diseased plant leaves
* Includes multiple classes such as:

  * Tomato diseases 🍅
  * Potato diseases 🥔
  * Pepper diseases 🌶️

---

## ⚙️ Model Details

* Base Model: **MobileNetV2 (pretrained on ImageNet)**
* Input Size: `224 x 224`
* Architecture:

  * MobileNetV2 (frozen layers)
  * Global Average Pooling
  * Dense Layer (ReLU)
  * Dropout
  * Output Layer (Softmax)

---

## 📈 Results

* ✅ Achieved ~**90%+ validation accuracy**
* 📊 Strong generalization with minimal overfitting
* 🔍 Evaluated using:

  * Accuracy & Loss Graphs
  * Confusion Matrix
  * Classification Report

---

## 🖥️ Application Workflow

1. Upload a leaf image
2. Image is preprocessed (resized + normalized)
3. Model predicts disease class
4. Output displayed with:

   * Predicted disease
   * Confidence score
   * Suggested remedy

---

## 📸 Demo

> Upload a plant leaf image and get instant predictions directly in the web app.

---

## 📁 Project Structure

```
Plant-Disease-App/
│
├── app.py
├── plant_disease_model.h5
├── class_names.json
├── requirements.txt
├── README.md
```

---

## ▶️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/plant-disease-detector.git

# Navigate to project folder
cd plant-disease-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🌐 Deployment

The application can be deployed using **Streamlit Cloud** for real-time access via a public URL.

---

## 🧠 Future Improvements

* 📷 Camera-based real-time detection
* 🌍 Deployment with custom domain
* 🤖 Integration with chatbot for plant care
* 📊 Top-3 predictions display
* 📱 Mobile app version

---

## 💡 Conclusion

This project demonstrates how deep learning can be applied in agriculture to assist in **early disease detection**, helping farmers and users take timely action.

---

## ⭐ If you found this useful, consider giving it a star!
