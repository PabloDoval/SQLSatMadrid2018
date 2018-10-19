import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import load_model

folderpath = './fashionMNIST'
filename = 'model'
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def score(image):
    model = load_model(folderpath, filename)
    image = preprocess_data(image)
    predictions = model.predict(image.reshape(1, image.shape[0], image.shape[1]))
    prediction = class_names[np.argmax(predictions[0])]
    return prediction

def preprocess_data(image):
    # We scale these values to a range of 0 to 1 before feeding to the neural network model.
    image = image / 255.0
    return image

if __name__ == '__main__': 
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    result = score(test_images[0])
    print(result)
