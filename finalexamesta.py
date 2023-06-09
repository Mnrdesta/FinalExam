import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True, hash_funcs={tf.keras.models.Model: id})
def load_model():
    model = tf.keras.models.load_model('Finals.h5')
    return model

def import_and_predict(image, model):
    size = (150, 150)  # Update the size to match the model's input shape
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

st.write("""
# Breeds Classifier
""")

file = st.file_uploader("Choose a dog photo from your computer", type=["jpg", "jpeg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    try:
        model = load_model()
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_names = ['yorkshire_terrier', 'poodle', 'golden_retriever', 'german_shepherd', 'french_bulldog']
        predicted_class = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class]
        st.success("OUTPUT: " + predicted_class_name)
    except Exception as e:
        st.error("Error: {}".format(e))
