import argparse

import warnings
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_hub as hub
import tensorflow_datasets as tfds
import logging
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str, default='./Capture.JPG',help = 'Enter file path') # use a deafault filepath to
    parser.add_argument('model',type = str,default = './new_model.h5', help = 'enter model path')
    parser.add_argument('k',type = int,default = 5, help = 'enter k')
    parser.add_argument('label_names', type = str, default='./label_map.json',help ='enter label path (json)')
    return parser.parse_args()

def load_split_data():
    
    dataset = tfds.load('oxford_flowers102', shuffle_files=True, as_supervised = True, with_info = False)
    
    train_set, test_set, val_set = dataset['train'] , dataset['test'], dataset['validation']
    num_training_examples = dataset_info.splits['train'].num_examples
    return training_set, test_set, valid_set, training_set, num_training_examples

image_size = 224 #Data is normalized and resized to 224x224 pixels as required by the pre-trained networks
def normalize(image, label):
    image = tf.cast(image, tf.float32) #from  unit8 to float32
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255 #rescaling images to be between 0 and 1
    return image, label

batch_size = 32 #I choose a smaller batch so that it can comfortably fit in my computer's memo
def batch_data(training_set, test_set, valid_set, num_training_examples):
    training_batches = training_set.cache().shuffle(num_training_examples//4).map(normalize(image, label)).batch(batch_size)
    test_batches = test_set.cache().shuffle(num_test_examples//4).map(normalize).batch(batch_size)
    valid_batches = valid_set.cache().shuffle(num_valid_examples//4).map(normalize).batch(batch_size)
    return training_batches, test_batches, valid_batches

def map_data(label):
    with open(label, 'r') as f:
        class_names = json.load(f)
    return class_names
model_use = 'new_model.h5'
def load_model():
    reloaded_keras_model = tf.keras.models.load_model(model_use, custom_objects={'KerasLayer':hub.KerasLayer})
    return reloaded_keras_model
   
def process_image(image):
    resized = tf.cast(image, tf.float32)
    resized = tf.image.resize(image, (224,224))
    resized /= 255
    resized = resized.numpy().squeeze()
    return resized

def predict(image_path, model, top_k,labels):
    im = Image.open(image_path)
    im = np.asarray(im)
    im = process_image(im) 
    im_final = np.expand_dims(im, axis = 0)
    p = model.predict(im_final)[0]
    probabilities = np.sort(p)[-top_k:len(p)]
    prbabilities = probabilities.tolist()
    classes = p.argpartition(-top_k)[-top_k:]
    top_classes = [labels[str(value+1)] for value in classes]
    probable_class = classes.tolist()
    print(len(probabilities), len(top_classes))
    return probabilities, top_classes


def main():
    args = parse_args()

    img_path = args.filepath
    model = args.model
    model = load_model()
    k = args.k
    label_S = args.label_names
    labels = map_data(label_S)
    probs, classes = predict(img_path, model, k, labels)
    probability = probs
    print('File selected: ' + img_path)
    print(classes)
    print(probability)
    lent = len(classes) - 1
    for i in range(len(classes)):
        print('{0} Prediction = {1} with probabilty {2}'.format(i+1, classes[lent - i], probability[lent - i]))
if __name__ == "__main__":
              main()
                                                      
                                                  
