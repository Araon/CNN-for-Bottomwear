# Image Classification using Convolutional Neural Networks (CNNs)

This project uses Convolutional Neural Networks (CNNs) to classify images based on their contents. The code is written in Python and uses the Keras library with TensorFlow backend.
This project is a part of the final year curriculum for NIFT students. It aims to demonstrate how to build an image classification model using a dataset of images and corresponding labels. The dataset used in this project is taken from a Google Sheet link and consists of images of different clothing items such as t-shirts, dresses, pants, etc.
# Data

The data used in this project is obtained from a Google Sheet link provided in the code. The link provides access to a CSV file containing image URLs and their corresponding labels. The images are downloaded using the URLs, resized to 100x100, and converted into a numpy array.
# Model Architecture

The model architecture consists of a single convolutional layer with 32 filters, each of size 3x3, followed by a max pooling layer with a pool size of 2x2. The output of the pooling layer is flattened and passed through two dense layers with 64 and 8 neurons respectively, and a softmax activation function is applied to the final layer to get a probability distribution over the classes.
# Training

The model is trained using the compiled categorical cross-entropy loss and Adam optimizer. The labels are converted to integer labels using the LabelEncoder class from scikit-learn and then converted to one-hot encoded vectors using the to_categorical function from Keras. The data is split into training and testing sets with a test size of 0.2, and the model is trained on the training set for 10 epochs with a batch size of 32.
# Evaluation

The model is evaluated on the test set using the evaluate function from Keras, and the accuracy is printed to the console.
# Dependencies

    pandas
    numpy
    Pillow
    requests
    tensorflow
    keras
    scikit-learn

# How to Run

    Clone the repository
    Install the dependencies
    Run the code in a Python environment with access to the internet

```python image_classification.py```
