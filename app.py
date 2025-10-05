import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ================================
# 🔹 Load the Trained Model
# ================================
@st.cache_resource
def load_model():
    try:
        # Try loading the .keras file first
        model = tf.keras.models.load_model("mobilenet.keras")
    except:
        # Fallback to .h5 if .keras not found
        model = tf.keras.models.load_model("mobilenet.h5")
    return model

model = load_model()

# ================================
# 🔹 Streamlit App UI
# ================================
st.set_page_config(page_title="Plastic vs E-Waste Classifier", layout="centered")

st.title("♻️ Plastic vs E-Waste Classifier (MobileNetV2)")
st.write("""
Welcome! This AI model classifies images as **Plastic Waste** or **E-Waste** using a MobileNetV2 deep learning model.  
📸 For the best results, open this app on your **mobile device** and use the **rear camera** under good lighting.
""")

# ================================
# 🔹 Image Input Options
# ================================
option = st.radio("Choose Input Method:", ["📁 Upload an Image", "📷 Use Camera"], horizontal=True)

image = None

if option == "📁 Upload an Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

elif option == "📷 Use Camera":
    captured_image = st.camera_input("Take a picture")
    if captured_image is not None:
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_container_width=True)

# ================================
# 🔹 Prediction Section
# ================================
if image is not None:
    st.subheader("🔍 Model Prediction")

    # Preprocess the image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    # Label based on threshold
    if confidence >= 0.5:
        label = "🟢 **PLASTIC WASTE**"
    else:
        label = "⚙️ **E-WASTE**"

    # Display results
    st.success(f"### ✅ Prediction: {label}")
    st.write(f"**Confidence Score:** {confidence:.4f}")

    # Add confidence progress bar
    st.progress(confidence if confidence < 1 else 1.0)

# ================================
# 🔹 About Section
# ================================
st.markdown("---")
st.markdown("""
### 📘 About this App
- **Model:** MobileNetV2 (Transfer Learning, ImageNet weights)
- **Input Size:** 128 × 128 RGB images  
- **Output Classes:** Plastic Waste / E-Waste  
- **Frameworks:** TensorFlow, Keras, Streamlit  
- **Developer:** Your Name Here  

💡 Tip: Use **well-lit, clear images** for better classification accuracy.
""")

