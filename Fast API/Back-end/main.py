from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("model.h5")


def index_to_word(integer, tokenizer):  # integer -> the index to be converted
    # tokenizer -> the object containing the word-to-index mapping
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


tokenizer_path = os.path.join(os.getcwd(), "tokenizer.pickle")
tokenizer = load_tokenizer(tokenizer_path)


#####
def load_feature(features_path):
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    return features


features_path = os.path.join(os.getcwd(), "features.pkl")
features = load_feature(features_path)


#####


def predict_caption(MODEL, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    # this determines the maximum length of generated captions).
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [sequence], max_length)
        # predict next word
        # represents probabilities across different words in vocabulary.
        yhat = MODEL.predict([image, sequence], verbose=0)
        # get index with high probability
        # The index with highest probability (argmax) is selected as the predicted word index.
        yhat = np.argmax(yhat)
        # convert index to word
        word = index_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


@app.post("/predict")
async def predict(file: UploadFile):
    # Save the uploaded file temporarily
    with open(file.filename, "wb") as image_file:
        image_file.write(file.file.read())

    # Extract the uploaded file's name without the extension
    filename_without_extension = os.path.splitext(file.filename)[0]

    # Now, you can open the saved image
    img_path = file.filename
    image = Image.open(img_path)

    # Use the filename_without_extension as the image_id
    image_id = filename_without_extension

    # Perform your prediction using the uploaded image
    y_pred = predict_caption(MODEL, features[image_id], tokenizer, 35)

    # Close the image file
    image.close()

    # Optionally, you can remove the temporary file
    os.remove(img_path)

    return {
        'class': y_pred,
        'confidence': 1
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=9000)
