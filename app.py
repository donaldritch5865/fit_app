import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("E:\\FINAL PROJ\\body_analysis\\bodytype_model.h5")
    
    return model

model = load_model()

# Class labels (must match training order)
class_names = ["Ectomorph", "Mesomorph", "Endomorph"]

st.title("🏋️‍♂️ Body Type Classifier")
st.write("Upload a full-body image to predict the body type.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (must match training preprocessing)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
