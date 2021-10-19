import numpy as np

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import backend as K

import tensorflow as tf

from keras.applications.resnet import ResNet50

# target model
base_model = ResNet50(include_top=True, weights='imagenet')
base_model.save("saved_model_resnet50")
