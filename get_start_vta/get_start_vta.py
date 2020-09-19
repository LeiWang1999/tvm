import os
import tvm
from tvm import te
import vta
import numpy as np
from tvm import rpc
from tvm.contrib import util
from vta.testing import simulator

# Load Vta parameters
env = vta.get_env()