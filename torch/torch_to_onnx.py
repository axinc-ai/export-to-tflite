# Export pytorch version of efficientnet lite

# load model
import timm
#model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
model = timm.create_model('efficientnet_lite0', pretrained=True)
model.eval()

# export to onnx
import torch
from torch.autograd import Variable
x = Variable(torch.randn(1, 3, 224, 224))
torch.onnx.export(model, x, 'efficientnetlite.onnx', verbose=True, opset_version=10)
