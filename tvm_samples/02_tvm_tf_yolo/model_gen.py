import tvm
import numpy as np
import tensorflow as tf
import tvm.relay.testing.tf as tf_testing
from tvm.contrib import graph_runtime
from tvm import relay
from PIL import Image
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Load Model
MODEL_PATH = "./model/yolov3_visdrone.pb"
with tf_compat_v1.gfile.GFile(MODEL_PATH, "rb") as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def,name="")
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, "pred_multi_scale/concat")

IMAGE_PATH = "./images/road.jpg"
image = Image.open(IMAGE_PATH, 'r').resize((416, 416))
x = np.array(image)
x = x[np.newaxis, ...]
shape_dict = {"input/input_data": x.shape}
mod, params = relay.frontend.from_tensorflow(graph_def, layout=None, shape=shape_dict)
print("Tensorflow protobuf imported to relay frontend.")

target = 'llvm'
target_host = 'llvm'
with tvm.transform.PassContext(opt_level=1):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)
ctx = tvm.cpu(0)
module = graph_runtime.GraphModule(lib['default'](ctx))
# set inputs
module.set_input("DecodeJpeg/contents", tvm.nd.array(x.astype(dtype)))
# execute
module.run()
# get outputs
tvm_output = module.get_output(0, tvm.nd.empty(((1, 1008)), "float32"))
print("Down")