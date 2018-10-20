import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.authentication import ServicePrincipalAuthentication

def get_workspace():
    # Security for the RBAC principal
    client_id = 'e0ae87de-e27b-433a-b7b5-017b0cd0808f'
    client_secret = 'de90f973-5a74-4e1c-a3a1-fa8265ca8ccd'
    tenant_id = '5c384fed-84cc-44a6-b34a-b060bf102a6e'
    servicePrincipalAuth = ServicePrincipalAuthentication(tenant_id, client_id, client_secret)
    workspace = Workspace.from_config(auth=servicePrincipalAuth)
    return workspace

def preprocess_data(image):
    image = image / 255.0
    image = image.reshape(1, image.shape[0], image.shape[1])
    return image

def init():
    global model
    azmodel = Model(get_workspace(), name='fashionMNIST')
    model_path = './model'
    model_name = os.path.join(model_path, 'fashionMNIST.h5')
    os.makedirs(model_path, exist_ok=True)
    if os.path.exists(model_name):
        os.remove(model_name)
    azmodel.download(target_dir=model_path, exists_ok=True)
    model = keras.models.load_model(model_name)

def run(image):
    try:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        image = preprocess_data(image)
        predictions = model.predict(image)
        result = class_names[np.argmax(predictions[0])]        
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result})

# if __name__ == '__main__': 
#     init()

#     fashion_mnist = keras.datasets.fashion_mnist
#     (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#     result = run(test_images[0])
#     print(result)
