import numpy as np
import sys
import cv2

import tensorflow as tf

# settings
input_model = "saved_model_resnet50"

output_model = "./models/output_quant.tflite"

# load validation set
import glob

image_folder = "../calibration"
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
    mean = np.asarray([103.939, 116.779, 123.68], dtype=np.float32)
    ary = ary - mean
    ary = np.minimum(ary,127)
    ary = np.maximum(ary,-128)
    validation_data_set.append(ary)

#quantize
def representative_dataset_gen():
  for i in range(len(validation_data_set)):
    yield [validation_data_set[i]]

converter = tf.lite.TFLiteConverter.from_saved_model(input_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

with open(output_model, 'wb') as o_:
    o_.write(tflite_quant_model)


