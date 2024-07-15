import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import seaborn as sns
from sklearn import metrics

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                                 r'Learning\real_vs_fake\real-vs-fake\train', target_size=(64, 64),
                                                 batch_size=32, class_mode='binary')
validation_set = test_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                                  'Learning\real_vs_fake\real-vs-fake\valid', target_size=(64, 64),
                                                  batch_size=32, class_mode='binary')
