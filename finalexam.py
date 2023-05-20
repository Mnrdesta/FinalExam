import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True, hash_funcs={tf.keras.models.Model: id})
def load_model():
    model = tf.keras.models.load_model('esta.h5')
    return model

def import_and_predict(image, model):
    size = (28, 28)
    image = ImageOps.fit(image, size, Image.ANTIALIAS).convert("L")  # Resize and convert to grayscale
    img = np.asarray(image) / 255.0  # Normalize pixel values
    img_reshape = img.reshape(1, 28, 28, 1)  # Reshape to match the model's input shape
    prediction = model.predict(img_reshape)
    return prediction

st.write("""
# Digits Classifier
""")

file = st.file_uploader("Choose digit photo from computer", type=["jpg", "jpeg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    try:
        model = load_model()
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        predicted_class = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class]
        st.success("OUTPUT: " + predicted_class_name)
    except Exception as e:
        st.error("Error: {}".format(e))
