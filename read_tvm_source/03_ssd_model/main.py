import tvm
from tvm import te

from matplotlib import pyplot as plt
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data, utils


def build(target):
    mod, params = relay.frontend.from_mxnet(block, {"data": dshape})
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    return lib


def run(lib, ctx):
    # Build TVM runtime
    m = graph_runtime.GraphModule(lib['default'](ctx))
    tvm_input = tvm.nd.array(x.asnumpy(), ctx=ctx)
    m.set_input('data', tvm_input)
    # execute
    m.run()
    # get outputs
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(
        1), m.get_output(2)
    return class_IDs, scores, bounding_boxs

# Load test Data
im_fname = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                             'gluoncv/detection/street_small.jpg?raw=true',
                             'street_small.jpg',
                             module='data')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

# Load supported_model ssd
supported_model = [
    'ssd_512_resnet50_v1_voc',
    'ssd_512_resnet50_v1_coco',
    'ssd_512_resnet101_v2_voc',
    'ssd_512_mobilenet1.0_voc',
    'ssd_512_mobilenet1.0_coco',
    'ssd_300_vgg16_atrous_voc'
    'ssd_512_vgg16_atrous_coco',
]

# Load Model
target = 'llvm'
model_name = supported_model[0]
dshape = (1, 3, 512, 512)
block = model_zoo.get_model(model_name, pretrained=True)
lib = build(target)
ctx = tvm.context(target)
class_IDs, scores, bounding_boxs = run(lib, ctx)
ax = utils.viz.plot_bbox(img,
                         bounding_boxs.asnumpy()[0],
                         scores.asnumpy()[0],
                         class_IDs.asnumpy()[0],
                         class_names=block.classes)
plt.show()