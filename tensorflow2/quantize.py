import numpy as np
import sys
import cv2
import glob

import tensorflow as tf

# settings
input_model = "saved_model"
output_model = "./models/output_quant.tflite"
image_folder = "../calibration"
#preprocess_mode = "caffe" # resnet50
preprocess_mode = "tf" # mobilenet, efficientnet

# load validation set
img_path = glob.glob(image_folder+"/*")
if len(img_path)==0:
    print("image not found")
    sys.exit(1)

#quantize
def representative_dataset_gen():
  for i in range(len(img_path)):
    print(img_path[i])
    img = cv2.imread(img_path[i]) #BGR
    img = cv2.resize(img,(224, 224))
    ary = np.asarray(img, dtype=np.float32)
    ary = np.expand_dims(ary, axis=0)
    if preprocess_mode == "caffe": # BGR
      mean = np.asarray([103.939, 116.779, 123.68], dtype=np.float32)
      ary = ary - mean
      ary = np.minimum(ary,127)
      ary = np.maximum(ary,-128)
    elif preprocess_mode == "tf": # RGB
      ary = ary[..., ::-1]
      ary = ary / 127.5
      ary = ary - 1.0
    else:
      raise "unknown preprocess mode"
    yield [ary]

converter = tf.lite.TFLiteConverter.from_saved_model(input_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

with open(output_model, 'wb') as o_:
    o_.write(tflite_quant_model)


