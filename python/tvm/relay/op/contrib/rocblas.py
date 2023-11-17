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
"""rocBLAS Relay integration."""
from typing import Callable, List, Tuple, Dict, Optional

import tvm
import tvm.ir
from tvm import relay
from tvm import te
from tvm.relay import transform
from tvm.contrib import rocblas

from ...dataflow_pattern import is_op, wildcard
from .te_target import lower_composite, relay_to_runtime
from .register import register_pattern_table


tvm._ffi.register_func("relay.ext.rocblas", relay_to_runtime(tvm.target.rocm()))


def partition_for_rocblas(
    mod: tvm.IRModule, params: Optional[Dict[str, tvm.runtime.NDArray]] = None
) -> tvm.IRModule:
    """Partition the graph to offload for rocBLAS.

    Parameters
    ----------
    mod : tvm.IRModule
        The module to partition.
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        Constant input parameters.

    Returns
    -------
    tvm.IRModule
        The partitioned module.
    """

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("rocblas"),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )
    return seq(mod)


@register_pattern_table("rocblas")
def pattern_table() -> List[Tuple[str, relay.Pattern, Callable[[relay.Call], bool]]]:
    """Get the rocBLAS pattern table."""

    def matmul_pattern() -> relay.Pattern:
        """Create pattern for matmul."""
        return is_op("nn.matmul")(wildcard(), wildcard())

    def batch_matmul_pattern() -> relay.Pattern:
        """Create pattern for batch_matmul."""
        return is_op("nn.batch_matmul")(wildcard(), wildcard())

    def dense_pattern() -> relay.Pattern:
        """Create pattern for dense."""
        return is_op("nn.dense")(wildcard(), wildcard())

    def check_matmul_like(matched: relay.Call) -> bool:
        """Check if matmul is supported by rocBLAS."""
        # Input data types can't be mixed
        if matched.args[0].checked_type.dtype != matched.args[1].checked_type.dtype:
            return False

        in_dtype = matched.args[0].checked_type.dtype
        out_dtype = matched.checked_type.dtype
        # Only the following data type combinations are supported
        if (in_dtype, out_dtype) not in [
            ("float32", "float32"),
            ("float16", "float16"),
            ("float16", "float32"),
            ("int8", "int32"),
            ("float64", "float64"),
            ("int8", "float32"),
        ]:
            return False

        # If inputs are int8, input column strides must be a multiple of 4
        if in_dtype == "int8":
            if (
                matched.args[0].checked_type.shape[-1] % 4 != 0
                or matched.args[1].checked_type.shape[-1] % 4 != 0
            ):
                return False

        return True

    return [
        ("rocblas.matmul", matmul_pattern(), check_matmul_like),
        ("rocblas.batch_matmul", batch_matmul_pattern(), check_matmul_like),
        ("rocblas.dense", dense_pattern(), check_matmul_like),
    ]


@lower_composite("rocblas.matmul")
def _lower_matmul(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a matmul using rocBLAS."""
    return rocblas.matmul(
        inputs[0],
        inputs[1],
        transa=op.attrs["transpose_a"],
        transb=op.attrs["transpose_b"],
    )


@lower_composite("rocblas.batch_matmul")
def _lower_batch_matmul(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a batch_matmul using rocBLAS."""
    return rocblas.batch_matmul(
        inputs[0],
        inputs[1],
        transa=op.attrs["transpose_a"],
        transb=op.attrs["transpose_b"],
    )


@lower_composite("rocblas.dense")
def _lower_dense(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a dense using rocBLAS."""
    return rocblas.matmul(
        inputs[0], inputs[1], transa=False, transb=True,
    )
