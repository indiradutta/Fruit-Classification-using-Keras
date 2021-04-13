import os
import matplotlib.pyplot as plt
from PIL import Image
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Create a generator for the training set
def preprocess():
  train_datagen = ImageDataGenerator(
    rescale=1./255
  )
  train_datagen = train_datagen.flow_from_directory(
          train,
          batch_size=32,
          target_size=(300, 300),
          class_mode='sparse')

  # Create a generator for the test set
  test_datagen = ImageDataGenerator(
    rescale=1./255
  )
  test_datagen = test_datagen.flow_from_directory(
          test,
          batch_size=32,
          target_size=(300, 300),
          class_mode='sparse')
  
  return train_datagen, test_datagen
