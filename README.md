# Image-Caption-generation-LSTM-CNN-

# Image Caption Generator with Flickr8k Dataset

This project is an implementation of an image caption generator using the Flickr8k dataset. The model generates captions for uploaded images by combining convolutional neural networks (CNN) and long short-term memory (LSTM) networks.

## Dataset

The [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k/download?datasetVersionNumber=1) dataset consists of 8,000 images sourced from the Flickr website. Each image in the dataset has multiple human-generated captions, providing rich textual descriptions.

To use this project, you will need to download and preprocess the Flickr8k dataset before training your model. Please refer to the `data_preprocessing.ipynb` notebook for detailed instructions on how to download and prepare the data.

## Model Architecture

The image caption generator uses a pre-trained Xception CNN as its encoder and an LSTM network as its decoder. The Xception CNN extracts visual features from input images, which are then fed into the LSTM network along with partial captions during training.

The architecture of this model is as follows:

- Input: Preprocessed Image
- Encoder: Pretrained Xception CNN
- Decoder: LSTM Network
    - Embedding Layer
    - Dropout Layer
    - LSTM Layers
    
For more details about the model architecture, please refer to `model.py`.

## Usage

1. Install dependencies by running `pip install -r requirements.txt`.
2. Download and preprocess the Flickr8k dataset using `data_preprocessing.ipynb`.
3. Train your own image captioning model using `train_model.ipynb`. Adjust hyperparameters if necessary.
4. Run Streamlit app locally with `streamlit run app.py` or deploy it to a web server.
5. Upload an image through Streamlit app and see the generated captions.

## Results

The model's performance is evaluated using the BLEU (bilingual evaluation understudy) metric, which measures the similarity between predicted and ground truth captions. The higher the BLEU score, the better the model performs.

During training, you can monitor loss and validation metrics to assess your model's progress. Additionally, you can visualize predictions using `evaluate_model.ipynb` to get a sense of how well your model generates captions for different images.

