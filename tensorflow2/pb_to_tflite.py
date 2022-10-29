import numpy as np
import tensorflow as tf

input_model = "saved_model"
output_model = "./models/output.tflite"

#to tensorflow lite
converter = tf.lite.TFLiteConverter.from_saved_model(input_model)
tflite_quant_model = converter.convert()
with open(output_model, 'wb') as o_:
    o_.write(tflite_quant_model)
