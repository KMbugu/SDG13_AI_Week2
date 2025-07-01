import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained MNIST model
model = tf.keras.models.load_model('mnist_cnn.h5')  # Assume model is saved from Task  Ascending
st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (28x28 grayscale)")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg"])

if uploaded_file is not None:
    # Preprocess image
    img = Image.open(uploaded_file).convert('L').resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Display image and prediction
    st.image(img, caption=f"Predicted Digit: {predicted_digit}", width=100)
    st.write(f"Confidence: {prediction[0][predicted_digit]:.2f}")

# Instructions: Run with `streamlit run mnist_streamlit.py`
# Save trained model as 'mnist_cnn.h5' from Task 2 before running