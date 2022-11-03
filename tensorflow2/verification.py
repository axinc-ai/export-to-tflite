import sys
import glob

import cv2
import numpy as np

from tensorflow.keras.applications.resnet import decode_predictions

import tensorflow as tf

# target model
output_model = "./models/output.tflite"
output_quant_model = "./models/output_quant.tflite"
#preprocess_mode = "caffe" # resnet50
preprocess_mode = "tf" # mobilenet

# load validation set
image_folder = "../calibration"
img_path = glob.glob(image_folder+"/*")
if len(img_path)==0:
    print("image not found")
    sys.exit(1)
max_n = 3
if len(img_path)>=max_n:
    img_path = img_path[0:max_n]

validation_data_set=[]
for file_name in img_path:
    img = cv2.imread(file_name) #BGR
    img = cv2.resize(img,(224, 224))
    ary = np.asarray(img, dtype=np.float32)
    ary = np.expand_dims(ary, axis=0)
    if preprocess_mode == "caffe": #BGR
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
    details = input_details[0]
    dtype = details['dtype']
    if dtype == np.uint8 or dtype == np.int8:
        quant_params = details['quantization_parameters']
        data = data / quant_params['scales'] + quant_params['zero_points']
        if dtype == np.int8:
            data = data.clip(-128, 127)
        else:
            data = data.clip(0, 255)
        data = data.astype(dtype)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])
    details = output_details[0]
    if details['dtype'] == np.uint8 or details['dtype'] == np.int8:
        quant_params = details['quantization_parameters']
        preds_tf_lite = preds_tf_lite - quant_params['zero_points']
        preds_tf_lite = preds_tf_lite.astype(np.float32) * quant_params['scales']
    print('Predicted:', decode_predictions(preds_tf_lite, top=3)[0])
