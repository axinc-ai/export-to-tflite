# Verify

# load model
import timm
#model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
model = timm.create_model('efficientnet_lite0', pretrained=True)
model.eval()

from torchvision import transforms
def trans(x):
    size = (224,224)
    x = transforms.Resize((size[0],size[1]))(x)
    x = transforms.ToTensor()(x)
    #print(x) # 0 - 1.0
    #x = x / 0.5 - 1
    x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
    return x


import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

#config = resolve_data_config({}, model=model)
#transform = create_transform(**config)

transform = trans

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension

# inference
import torch
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)

# Get imagenet class mappings
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
print("torch")
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

# inference
import ailia_tflite
#import tensorflow as tf

interpreter = ailia_tflite.Interpreter(model_path="saved_model/model_float32_opt.tflite")
#interpreter = tf.lite.Interpreter(model_path="saved_model/model_float32.tflite")
interpreter.allocate_tensors()
#print(tensor.shape)
#print(interpreter.get_input_details())
input_data = tensor.numpy().transpose((0,2,3,1))
#print(input_data.shape)
interpreter.set_tensor(0, input_data)
interpreter.invoke()
probabilities = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
probabilities = torch.from_numpy(probabilities)
#print(probabilities.shape)
probabilities = torch.nn.functional.softmax(probabilities[0], dim=0)

print("ailia tflite")
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
