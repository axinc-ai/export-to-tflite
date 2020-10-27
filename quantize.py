import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import sys
import cv2

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import backend as K
import numpy as np
import tensorflow as tf

# settings
input_model = "./models/output.pb"
input_name = "model_input"
output_node_name = "mobilenetv2_1.00_224/Logits/Softmax"

output_model = "./models/output_quant.tflite"

# load validation set
import glob

image_folder = "./calibration"
img_path = glob.glob(image_folder+"/*")
if len(img_path)==0:
    print("image not found")
    sys.exit(1)

validation_data_set=[]
for file_name in img_path:
    img = cv2.imread(file_name) #BGR
    img = cv2.resize(img,(224, 224))
    ary = np.asarray(img, dtype=np.float32)
    ary = np.expand_dims(ary, axis=0)
    ary = ary/255.0
    validation_data_set.append(ary)

#quantize
def representative_dataset_gen():
  for i in range(len(validation_data_set)):
    yield [validation_data_set[i]]

converter = tf.lite.TFLiteConverter.from_frozen_graph(input_model,[input_name],[output_node_name])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open(output_model, 'wb') as o_:
    o_.write(tflite_quant_model)


