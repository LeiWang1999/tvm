# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument
"""MIOPEN Relay integration."""
from typing import Callable, List, Tuple

import tvm
import tvm.ir
from tvm import relay
from tvm import te
from tvm.relay import transform
from tvm.contrib import miopen

from ...dataflow_pattern import is_op, wildcard
from .te_target import lower_composite, relay_to_runtime
from .register import register_pattern_table


tvm._ffi.register_func("relay.ext.miopen", relay_to_runtime(tvm.target.rocm()))


def partition_for_miopen(mod: tvm.IRModule) -> tvm.IRModule:
    """Partition the graph to offload for MIOPEN.

    Parameters
    ----------
    mod : tvm.IRModule
        The module to partition.

    Returns
    -------
    tvm.IRModule
        The partitioned module.
    """

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("miopen"),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )
    return seq(mod)


@register_pattern_table("miopen")
def pattern_table() -> List[Tuple[str, relay.Pattern, Callable[[relay.Call], bool]]]:
    """Get the MIOPEN pattern table."""

    def softmax_pattern() -> relay.Pattern:
        """Create pattern for softmax."""
        return is_op("nn.softmax")(wildcard())

    def log_softmax_pattern() -> relay.Pattern:
        """Create pattern for log_softmax."""
        return is_op("nn.log_softmax")(wildcard())

    def conv2d_pattern() -> relay.Pattern:
        """Create pattern for conv2d."""
        return is_op("nn.conv2d")(wildcard(), wildcard())

    def conv2d_bias_act_pattern() -> relay.Pattern:
        """Create pattern for fused conv2d+bias+activation."""
        conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
        bias = is_op("nn.bias_add")(conv2d, wildcard())
        return bias.optional(is_op("nn.relu"))

    def check_softmax(matched: relay.Call) -> bool:
        """Check if softmax is supported by MIOPEN."""
        if matched.args[0].checked_type.dtype not in ["float64", "float32", "float16"]:
            return False

        return True

    def check_log_softmax(matched: relay.Call) -> bool:
        """Check if log_softmax is supported by MIOPEN."""
        if matched.args[0].checked_type.dtype not in ["float64", "float32", "float16"]:
            return False

        if len(matched.args[0].checked_type.shape) != 2:
            return False

        if matched.attrs["axis"] not in (1, -1):
            return False

        return True

    def check_conv2d(matched: relay.Call) -> bool:
        if matched.args[0].checked_type.dtype not in ["float64", "float32", "float16"]:
            return False

        if matched.attrs["data_layout"] != "NCHW" or matched.attrs["kernel_layout"] != "OIHW":
            return False

        padding = matched.attrs["padding"]
        if padding[0] != padding[2] or padding[1] != padding[3]:
            return False

        return True

    def check_conv2d_bias_act(matched: relay.Call) -> bool:
        return True

    return [
        ("miopen.softmax", softmax_pattern(), check_softmax),
        ("miopen.log_softmax", log_softmax_pattern(), check_log_softmax),
        ("miopen.conv2d", conv2d_pattern(), check_conv2d),
    ]


@lower_composite("miopen.softmax")
def _lower_softmax(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a softmax using MIOPEN."""
    return miopen.softmax(inputs[0], axis=op.attrs["axis"])


@lower_composite("miopen.log_softmax")
def _lower_log_softmax(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a log_softmax using MIOPEN."""
    return miopen.log_softmax(inputs[0], axis=op.attrs["axis"])


@lower_composite("miopen.conv2d")
def _lower_conv2d(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a conv2d using MIOPEN."""
    pad_h, pad_w = op.attrs["padding"][:2]
    stride_h, stride_w = op.attrs["strides"][:2]
    dilation_h, dilation_w = op.attrs["dilation"][:2]
    group_count = op.attrs["groups"]
    conv_mode = 0
    dtype = op.checked_type.dtype
    if dtype == "float16":
        conv_dtype = 0
    elif dtype == "float32":
        conv_dtype = 1
    print("type of inputs[0]: ", type(inputs[0]))
    print("type of inputs[1]: ", type(inputs[1]))
    print("type of pad_h: ", type(pad_h))
    print("type of pad_w: ", type(pad_w))
    print("type of stride_h: ", type(stride_h))
    print("type of stride_w: ", type(stride_w))
    print("type of dilation_h: ", type(dilation_h))
    print("type of dilation_w: ", type(dilation_w))
    print("type of conv_mode: ", type(conv_mode))
    print("type of conv_dtype: ", type(conv_dtype))
    print("type of group_count: ", type(group_count))
    
    return miopen.conv2d_forward(
        inputs[0],
        inputs[1],
        int(pad_h),
        int(pad_w),
        int(stride_h),
        int(stride_w),
        int(dilation_h),
        int(dilation_w),
        conv_mode=conv_mode,
        data_type=conv_dtype,
        group_count=group_count
    )
