import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/cnn_model.h5")

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

st.title("CNN Image Classifier 🚀")

st.write("Upload an image and the model will predict it.")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((32,32))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    st.subheader("Prediction:")
    st.write(classes[class_index])