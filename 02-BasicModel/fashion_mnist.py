import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def display_images(train_images, train_labels, class_names, num_images=25):
    plt.figure(figsize=(10,10))
    for i in range(num_images):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

def plot_images_and_predictions(predictions, test_images, test_labels, num_rows=5, num_cols=3):
    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

def load_data():
    # Load fashion mnist dataset
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

def get_model_definition():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model

def train_model(model, train_images, train_labels, epochs=5):
    # Compile model
    model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train model
    model.fit(train_images, train_labels, epochs=epochs)    

def evaluate(model, test_images, test_labels):   
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

def predict(model, test_images, test_labels, class_names):
    predictions = model.predict(test_images)

    print('Predictions for first image:')
    print(predictions[0])
    print('Category predicted for first image: ' + class_names[np.argmax(predictions[0])])
    print('Real category for first image: ' + class_names[test_labels[0]])

    plot_images_and_predictions(predictions, test_images, test_labels)

if __name__ == '__main__': 
    # Load data
    train_images, train_labels, test_images, test_labels, class_names = load_data()

    # Explore data
    print(train_images.shape)
    print(train_labels)
    print(test_images.shape)
    display_images(train_images, train_labels, class_names)

    # Preprocess data
    train_images, test_images = preprocess_data(train_images, test_images)

    # Build model
    model = get_model_definition()

    # Train model
    train_model(model, train_images, train_labels)

    # Evaluate accuracy
    evaluate(model, test_images, test_labels)

    # Predict test images
    predict(model, test_images, test_labels, class_names)


