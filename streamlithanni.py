import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('hanni_classifier_model.h5')

# Image size your model expects
IMG_SIZE = 128

# Function to preprocess the uploaded image
def preprocess_image(image, img_size):
    # Convert the image to RGB (if it has an alpha channel or is grayscale)
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Convert the image to a NumPy array
    img_array = np.array(image)
    # Resize the image to the model's expected size
    img_resized = cv2.resize(img_array, (img_size, img_size))
    # Normalize the image (scale pixel values between 0 and 1)
    img_resized = img_resized / 255.0
    # Add a batch dimension
    return np.expand_dims(img_resized, axis=0)

# Streamlit app
st.title("Hanni Classifier")
st.write("Upload an image to classify whether it is of Hanni or not.")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image and make predictions
    with st.spinner("Classifying..."):
        preprocessed_image = preprocess_image(image, IMG_SIZE)
        predictions = model.predict(preprocessed_image)
        label = np.argmax(predictions)  # Get the predicted class
        confidence = predictions[0][label]  # Get the confidence score

    # Display the classification result
    if label == 0:
        st.success(f"This is Hanni! (Confidence: {confidence:.2f})")
    else:
        st.error(f"This is NOT Hanni. (Confidence: {confidence:.2f})")
