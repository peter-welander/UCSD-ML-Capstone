import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

from PIL import Image

@st.cache_resource()
def load_model():
	model = tf.keras.models.load_model('saved_model/final')
	return model

@st.cache_resource()
def load_class_names():
    class_names = []
    with open('saved_model/flowers_final_labels.pickle', "rb") as f:
        class_names = pickle.loads(f.read())
    return class_names

def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [192, 192])
	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction

class_names = load_class_names()
model = load_model()

st.title('Flower Classifier')
file = st.file_uploader("Upload an image of a flower", type=["jpg", "png"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = Image.open(file)
	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)


	result = class_names[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)
