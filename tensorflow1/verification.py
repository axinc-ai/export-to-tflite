import sys
import glob

import cv2
import numpy as np

from keras.applications.resnet50 import decode_predictions

import tensorflow as tf

# target model
output_model = "./models/output.tflite"
output_quant_model = "./models/output_quant.tflite"

# load validation set
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
    ary = ary/255.0
    validation_data_set.append(ary)

#verify Tensorflow lite
interpreter = tf.lite.Interpreter(model_path=output_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("tf lite float")

for i in range(len(validation_data_set)):
    interpreter.set_tensor(input_details[0]['index'], validation_data_set[i])
    interpreter.invoke()
    preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])
    print('Predicted:', decode_predictions(preds_tf_lite, top=3)[0])

#verify quantized tensorflow lite
interpreter = tf.lite.Interpreter(model_path=output_quant_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("tf lite int8")

for i in range(len(validation_data_set)):
    data = validation_data_set[i]
    data = data*255
    data = data.astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])
    print('Predicted:', decode_predictions(preds_tf_lite, top=3)[0])
