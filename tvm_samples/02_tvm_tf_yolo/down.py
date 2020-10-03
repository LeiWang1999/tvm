# tvm, relay
import tvm
from tvm import te
from tvm import relay

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

# Base location for model related files.
repo_base = "https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/"

# Test image
img_name = "elephant-299.jpg"
image_url = os.path.join(repo_base, img_name)

model_name = "classify_image_graph_def-with_shapes.pb"
model_url = os.path.join(repo_base, model_name)

# Image label map
map_proto = "imagenet_2012_challenge_label_map_proto.pbtxt"
map_proto_url = os.path.join(repo_base, map_proto)

# Human readable text for labels
label_map = "imagenet_synset_to_human_label_map.txt"
label_map_url = os.path.join(repo_base, label_map)

# Target settings
# Use these commented settings to build for cuda.
# target = 'cuda'
# target_host = 'llvm'
# layout = "NCHW"
# ctx = tvm.gpu(0)
target = "llvm"
target_host = "llvm"
layout = None
ctx = tvm.cpu(0)

from tvm.contrib.download import download_testdata

img_path = download_testdata(image_url, img_name, module="data")
model_path = download_testdata(model_url, model_name, module=["tf", "InceptionV1"])
map_proto_path = download_testdata(map_proto_url, map_proto, module="data")
label_path = download_testdata(label_map_url, label_map, module="data")