import numpy as np

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import backend as K

import tensorflow as tf

# target model

#ResNet50 : Preprocess imagenet_utils.preprocess_input(x, data_format=data_format, mode="caffe") BGR
#from tensorflow.keras.applications.resnet import ResNet50
#base_model = ResNet50(include_top=True, weights='imagenet')

#MobileNet : Preprocess imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf") RGB
#from tensorflow.keras.applications.mobilenet import MobileNet
#base_model = MobileNet(include_top=True, weights='imagenet')

#MobileNetV2 : Preprocess imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf") RGB
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#base_model = MobileNetV2(include_top=True, weights='imagenet')

#EfficientNetLite : Preprocess imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf") RGB
from efficientnet_lite import EfficientNetLiteB0
import tensorflow as tf
base_model = EfficientNetLiteB0(weights='imagenet', input_shape=(224, 224, 3))

base_model.save("saved_model")
