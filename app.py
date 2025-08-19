import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# ------------------------------
# Load trained model
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("/Users/donald/fit_app/bodytype_model.h5")
    return model

model = load_model()

# Class labels (must match training order)
class_names = ["Ectomorph", "Mesomorph", "Endomorph"]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🏋️‍♂️ Body Type Classifier")
st.write("Upload a full-body image to predict the body type.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Safely read the uploaded file
    try:
        # Convert uploaded file -> bytes -> Image
        image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
        uploaded_file.seek(0)  # reset pointer so file can be reused
    except Exception as e:
        st.error(f"❌ Could not open image: {e}")
        st.stop()

    # Show uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ------------------------------
    # Preprocess image
    # ------------------------------
    img = image.resize((224, 224))   # resize to model input size
    img_array = np.array(img) / 255.0  # normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)

    # ------------------------------
    # Prediction
    # ------------------------------
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("🔎 Results")
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # ------------------------------
    # Show probabilities for all classes
    # ------------------------------
    st.subheader("📊 Class Probabilities")

    # Display as table
    for i, class_name in enumerate(class_names):
        st.write
