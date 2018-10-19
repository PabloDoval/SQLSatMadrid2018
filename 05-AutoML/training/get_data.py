from tensorflow import keras
import numpy as np

def get_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return train_images, train_labels, test_images, test_labels, class_names

    return { "X" : train_images, "y" : train_labels }