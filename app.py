#app.py
import io
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from utils import clean_image, get_prediction, make_results
@st.cache_resource
def load_model_cached():
    """Load and cache the trained TensorFlow model."""
    try:
        model = load_model("model.h5")
        st.sidebar.success(" Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f" Error loading model: {e}")
        return None

def clean_image(image, target_size=(225, 225)):
    """Preprocess the image to match the model input shape."""
    image = image.convert("RGB")                # Ensure 3 channels (RGB)
    image = image.resize(target_size, Image.LANCZOS)  # Resize
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)       # Add batch dimension
    return image

def get_prediction(model, image):
    """Generate predictions from the model."""
    predictions = model.predict(image)
    return predictions, np.argmax(predictions, axis=1)

def make_results(predictions, predictions_arr):
    labels = ["Healthy", "Multiple Diseases", "Rust", "Scab"]
    result = {
        "status": labels[int(predictions_arr[0])],
        "prediction": f"{predictions[0][predictions_arr[0]] * 100:.2f}%"
    }
    return result, labels, predictions

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# Hide Streamlit menu/footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load model once
model = load_model_cached()
input_height, input_width = (225, 225)

st.title("Plant Disease Detection")
st.write("Upload an image of a plant leaf to detect possible diseases.")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if model is None:
        st.error("Model not loaded. Please check 'model.h5' path.")
    else:
        try:
            # Load and display uploaded image
            image = Image.open(io.BytesIO(uploaded_file.read()))
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Preprocess image
            processed_image = clean_image(image, target_size=(input_width, input_height))
            st.write(f"Processed image shape: {processed_image.shape}")

            # Convert preprocessed array to displayable format (safe for grayscale/RGBA)
            display_img = processed_image[0]
            if display_img.ndim == 2:  # grayscale fallback
                display_img = np.stack([display_img] * 3, axis=-1)
            elif display_img.shape[-1] > 3:  # RGBA/CMYK fallback
                display_img = display_img[..., :3]

            display_img = (display_img * 255).astype(np.uint8)
            st.image(display_img, caption="Processed Image", use_container_width=True)

            # Make predictions
            with st.spinner("Running inference..."):
                predictions, predictions_arr = get_prediction(model, processed_image)

            # Format results
            result, labels, predictions = make_results(predictions, predictions_arr)

            # Display probabilities
            st.write("Prediction Probabilities:")
            for i, label in enumerate(labels):
                st.write(f"- **{label}**: {predictions[0][i] * 100:.2f}%")

            # Final result
            st.success(f" The plant is **{result['status']}** "
                       f"with **{result['prediction']}** confidence.")

        except Exception as e:
            st.error(f"Error processing image: {e}")
else:
    st.info("Upload a plant leaf image to begin analysis.")
