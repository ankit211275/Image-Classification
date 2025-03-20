import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Load the trained model
model = load_model("models/model.h5", compile=False)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Define class labels
class_labels = ["Happy", "Sad"]

# Function to preprocess images
def preprocess_image(image):
    img = image.convert("RGB")  # Convert image mode if necessary
    img = img.resize((256, 256))  # Resize to model's expected input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make predictions
def predict_image(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    return "Sad" if prediction > 0.5 else "Happy"

# Streamlit App
st.set_page_config(page_title="Image Classification", layout="wide")

# Custom CSS to adjust camera input size
st.markdown("""
    <style>
        div[data-testid="stCameraInput"] {
            width: 500px !important;  
            height: auto !important; /* Auto height to fit content */
            margin: auto; 
            padding: 0px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            overflow: hidden !important;
        }
        video {
            width: 500px !important; 
            height: 400px !important;
            object-fit: cover !important;
        }
        button[data-testid="stCameraCaptureButton"] {
            width: 100% !important; 
            height: 50px !important; 
            font-size: 16px !important; 
            border-radius: 8px !important;
            margin-top: 10px !important;
            display: block !important;
        }
    </style>
""", unsafe_allow_html=True)


# Initialize session state for captured image
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

st.title("🎭 Image Classification: Happy or Sad")
st.write("Upload an image or capture a live photo to classify.")

# Choice for image input method
option = st.radio("Select Input Method:", ["Upload Image", "Take a Picture"], key="input_method")

uploaded_file = None
captured_image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="file_uploader")

elif option == "Take a Picture":
    if st.session_state.captured_image is None:  # Show camera input only if no image is captured
        captured_image = st.camera_input("Take a Picture")

    # If an image is captured, store it in session state
    if captured_image is not None and st.session_state.captured_image is None:
        st.session_state.captured_image = captured_image  # Save the image in session state

# Display the selected or captured image
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)

elif st.session_state.captured_image is not None:  # Use session state image if available
    image = Image.open(st.session_state.captured_image)

if image:
    col1, col2 = st.columns([0.5, 1.5])

    with col1:
        st.image(image, caption="Selected Image", use_container_width=True)

    if st.button("Predict", key="predict_button"):
        with st.spinner("🔄 Predicting..."):
            time.sleep(2)  # Simulating loading time
            label = predict_image(image)

        with col2:
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; height: 250px;">  
                <div style="flex: 1; display: flex; align-items: center; justify-content: center; 
                            background-color: {'lightgreen' if label == 'Happy' else '#2222'}; 
                            color: white; padding: 20px; font-size: 24px; min-height: 80px;">
                    😊 Happy
                </div>
                <div style="flex: 1; display: flex; align-items: center; justify-content: center; 
                            background-color: {'coral' if label == 'Sad' else '#2222'}; 
                            color: white; padding: 20px; font-size: 24px; min-height: 80px;">
                    😢 Sad
                </div>
            </div>
            """, unsafe_allow_html=True)

# Reset button to allow taking another picture
if option == "Take a Picture" and st.session_state.captured_image is not None:
    if st.button("Retake Photo"):
        st.session_state.captured_image = None  # Reset captured image
