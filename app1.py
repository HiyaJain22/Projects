import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model_path = 'C:/Users/hjain/Downloads/Food_dataset/saved_model.h5'  # Replace with the path to your trained model file
model = load_model(model_path)

# Define your class labels here
class_labels = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']  # Replace with your actual class labels

st.title("Food Image Classifier")

# Upload image through Streamlit
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Load and preprocess the uploaded image
    img = load_img(uploaded_image, target_size=(150, 150))
    x = img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)

    # Make predictions
    predictions = model.predict(x)
    class_index = np.argmax(predictions[0])
    predicted_class = class_labels[class_index]

    st.write(f"Predicted Class: {predicted_class}")
