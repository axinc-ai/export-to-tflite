import numpy as np

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import backend as K

import tensorflow as tf

from keras.applications.mobilenet_v2 import MobileNetV2
#from keras.models import load_model

# target model
base_model = MobileNetV2(include_top=True, weights='imagenet')

# target model from hdf5
#base_model = load_model("input.hdf5")

# model input and output
input_shape = [1, 224, 224, 3]
input_node_name = "mobilenetv2_1.00_224/Conv1_pad/Pad"
output_node_name = "mobilenetv2_1.00_224/Logits/Softmax"

output_folder = "./models"
output_name = "output.pb"

# export from tensorflow
tf.keras.backend.set_learning_phase(0)

x = tf.placeholder(tf.float32, input_shape, name="model_input")
y = base_model(x)
base_model.summary()

# trainable & uninitialized variables
uninitialized_variables = [v for v in tf.global_variables() \
    if not hasattr(v, '_keras_initialized') or not v._keras_initialized]

# initialization
sess = K.get_session()

gd = sess.graph.as_graph_def()

print("node list")
for node in gd.node:
    print(node)

sess.run(tf.variables_initializer(uninitialized_variables))

frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess,gd,[output_node_name])
tf.train.write_graph(frozen_graph_def,output_folder,name=output_name,as_text=False)
