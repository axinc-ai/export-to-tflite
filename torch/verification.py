# Verify

import torch
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
import numpy as np

# settings
use_ailia = False
use_official_preprocess = False

if use_ailia:
    import ailia_tflite
else:
    import tensorflow as tf

# load torch model
#model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
model = timm.create_model('efficientnet_lite0', pretrained=True)
model.eval()

# preprocess
def original_trans(x):
    size = (224,224)
    x = transforms.Resize((size[0],size[1]))(x)
    x = transforms.ToTensor()(x)    # 0-255 -> 0-1
    #x = x / 0.5 - 1    # tf_efficientnet_lite0
    x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)  # efficientnet_lite0
    return x

if use_official_preprocess:
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
else:
    transform = original_trans

# load test image
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension

# Get imagenet class mappings
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# torch
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print("torch")
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

# tflite (float)
print("tflite (float)")
if use_ailia:
    interpreter = ailia_tflite.Interpreter(model_path="saved_model/model_float32_opt.tflite")
else:
    interpreter = tf.lite.Interpreter(model_path="saved_model/model_float32.tflite")
interpreter.allocate_tensors()
input_data = tensor.numpy().transpose((0,2,3,1))
interpreter.set_tensor(0, input_data)
interpreter.invoke()
probabilities = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
probabilities = torch.from_numpy(probabilities)
probabilities = torch.nn.functional.softmax(probabilities[0], dim=0)

top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

# tflite (int8)
print("tflite (int8)")
if use_ailia:
    interpreter = ailia_tflite.Interpreter(model_path="efficientnetlite_quant.tflite")
else:
    interpreter = tf.lite.Interpreter(model_path="efficientnetlite_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
details = input_details[0]
dtype = details['dtype']
if dtype == np.uint8 or dtype == np.int8:
    quant_params = details['quantization_parameters']
    data = input_data / quant_params['scales'] + quant_params['zero_points']
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
probabilities = torch.from_numpy(preds_tf_lite)
probabilities = torch.nn.functional.softmax(probabilities[0], dim=0)

top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
