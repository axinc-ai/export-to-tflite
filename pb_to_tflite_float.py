import numpy as np
import tensorflow as tf

input_model = "./models/output.pb"
input_name = "model_input"
output_node_name = "mobilenetv2_1.00_224/Logits/Softmax"

output_model = "./models/output.tflite"

#to tensorflow lite
converter = tf.lite.TFLiteConverter.from_frozen_graph(input_model,[input_name],[output_node_name])
tflite_quant_model = converter.convert()
with open(output_model, 'wb') as o_:
    o_.write(tflite_quant_model)
