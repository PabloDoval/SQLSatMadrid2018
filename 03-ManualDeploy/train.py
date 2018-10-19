import tensorflow as tf
from tensorflow import keras
from model import get_model_definition, save_model
from train import train_model

def train_model(model, train_images, train_labels, epochs=5):
    model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=epochs)    

def evaluate(model, test_images, test_labels):   
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return train_images, train_labels, test_images, test_labels, class_names

def preprocess_data(train_images, test_images):
    # Preprocess the data
    # The data must be preprocessed before training the network. 
    # The pixel values fall in the range of 0 to 255
    # We scale these values to a range of 0 to 1 before feeding to the neural network model.
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, test_images

if __name__ == '__main__': 
    # Load data
    train_images, train_labels, test_images, test_labels, class_names = load_data()

    # Preprocess data
    train_images, test_images = preprocess_data(train_images, test_images)

    # Build model
    model = get_model_definition()

    # Train model
    train_model(model, train_images, train_labels)

    # Save model
    folderpath = './fashionMNIST'
    filename = 'model'
    save_model(model, folderpath, filename)
