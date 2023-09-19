import streamlit as st
from keras.models import Model
from keras.preprocessing import image
from keras.applications.xception import Xception, preprocess_input
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import load_img, img_to_array
from PIL import Image
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
import requests
from io import BytesIO

st.title("Image Captioning")
st.write("Upload your image or provide an image URL to get its caption!")

model = Xception()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
caption_model = load_model('D:\Array AI Diploma\Week 26\Final Project\model.h5')
tokenizer = Tokenizer()
with open('D:\Array AI Diploma\Week 26\Final Project\\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    

def idx_to_word(index, tokenizer):
    for word, i in tokenizer.word_index.items():
        if i == index:
            return word
    return None

def predict_caption(image, max_length=35):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        prediction = caption_model.predict([ image, sequence ], verbose=0)
        prediction = np.argmax(prediction)
        word = idx_to_word(prediction,tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    in_text = in_text.replace('startseq', '').replace('endseq', '')
    return in_text.strip()

option = st.radio("Choose an image source:", ("Upload an image", "Provide an image URL"))

if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        img = img.resize((299, 299))
        img = img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        feature = model.predict(img, verbose=0)
        prediction = predict_caption(feature)
        st.write("Caption : ", prediction)

else:
    image_url = st.text_input("Enter the image URL:")

    if st.button("Get Caption") and image_url:
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                st.image(img, caption='Image from URL', use_column_width=True)
                img = img.resize((299, 299))
                img = img_to_array(img)
                img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
                img = preprocess_input(img)
                feature = model.predict(img, verbose=0)
                prediction = predict_caption(feature)
                st.write("Caption : ", prediction)
            else:
                st.write("Error: Unable to fetch the image from the URL.")
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")
