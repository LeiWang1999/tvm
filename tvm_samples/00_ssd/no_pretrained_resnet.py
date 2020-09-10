import os
import tvm
from tvm import relay
from tvm.relay import testing
import numpy as np
import time
from tvm.contrib import graph_runtime
from PIL import Image

# Image Processing
def image_preprocessing(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image.astype('float32')

image_path = "./cat.jpeg"
dsize = (1, 3, 224, 224)
raw_image=Image.open(image_path).resize((224, 224))
images = image_preprocessing(raw_image)
label_path = './imagenet1k_labels.txt'
with open(label_path) as f:
    labels = eval(f.read())

# Load Model
base = './'
path_lib = base + 'model.so'
path_graph = base + 'model.json'
path_param = base + 'model.params'

batch_size = 1
num_classes = 1000
load_module = False
print(images.shape)
ctx = tvm.cpu()
if load_module:
    lib = tvm.module.load(path_lib)
    graph = open(path_graph).read()
    params = bytearray(open(path_param, 'rb').read())
    module = graph_runtime.create(graph, lib, ctx)
    module.load_params(params)
else:
    model, params = tvm.relay.testing.resnet.get_workload()
    opt_level = 3
    target = 'llvm'
    with tvm.transform.PassContext(opt_level=opt_level):
        graph, lib, params = relay.build(model, target, params=params)
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input('data', images)
    module.set_input(**params)

module.run()
scores = module.get_output(0).asnumpy()[0]
print(scores.shape)
out = np.argsort(scores)[-1:-5:-1]
print(labels[out[0]], labels[out[1]])




