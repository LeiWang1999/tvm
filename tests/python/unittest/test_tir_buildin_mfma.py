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

import sys
import pytest

import tvm
from tvm.script import tir as T
import numpy as np
import tvm.testing


@T.prim_func
def gemm_mfma_m16n16k4_row_row_fp32pf32fp32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 4], dtype="float32")
    B = T.match_buffer(b, [4, 16], dtype="float32")
    C = T.match_buffer(c, [16, 16], dtype="float32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 64)
    MultiA = T.decl_buffer([1], "float32", scope="local")
    MultiB = T.decl_buffer([1], "float32", scope="local")
    Accum = T.decl_buffer([4], "float32", scope="local")
    for i in range(4):
        Accum[i] = T.float64(0)

    MultiA[0] = A[tx % 16, tx // 16]
    MultiB[0] = B[tx // 16, tx % 16]
    T.evaluate(T.tvm_mfma(
        "f32_16x16x4f32",
        "row",
        "row",
        "float32",
        "float32",
        "float32x4",
        MultiA.data,
        0,
        MultiB.data,
        0,
        Accum.data,
        0,
        dtype="float32x4",
    ))
    for mma_accum_c_id in range(4):
        C[tx // 16 * 4 + mma_accum_c_id, tx % 16] = Accum[mma_accum_c_id]


@tvm.testing.requires_rocm
def test_gemm_mfma_m16n16k4_row_row_fp32pf32fp32():
    sch = tvm.tir.Schedule(gemm_mfma_m16n16k4_row_row_fp32pf32fp32)
    hip_mod = tvm.build(sch.mod, target="hip")
    print(hip_mod.imported_modules[0].get_source())
    A_np = np.random.uniform(-1, 1, [16, 4]).astype("float32")
    B_np = np.random.uniform(-1, 1, [4, 16]).astype("float32")
    C_np = np.zeros([16, 16]).astype("float32")

    ctx = tvm.rocm()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    hip_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("float32"), B_np.astype("float32"))

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mfma_m16n16k16_row_row_fp16fp16fp32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 16], dtype="float16")
    B = T.match_buffer(b, [16, 16], dtype="float16")
    C = T.match_buffer(c, [16, 16], dtype="float32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 64)
    MultiA = T.decl_buffer([4], "float16", scope="local")
    MultiB = T.decl_buffer([4], "float16", scope="local")
    Accum = T.decl_buffer([4], "float32", scope="local")

    for i in range(4):
        Accum[i] = T.float64(0)
    for i in range(4):
        MultiA[i] = A[tx % 16, tx // 16 * 4 + i]
    for i in range(4):
        MultiB[i] = B[i + tx // 16 * 4, tx % 16]

    T.evaluate(T.tvm_mfma(
        "f32_16x16x16f16",
        "row",
        "row",
        "float16x4",
        "float16x4",
        "float32x4",
        MultiA.data,
        0,
        MultiB.data,
        0,
        Accum.data,
        0,
        dtype="float32x4",
    ))
    for mma_accum_c_id in range(4):
        C[tx // 16 * 4 + mma_accum_c_id, tx % 16] = Accum[mma_accum_c_id]


@tvm.testing.requires_rocm
def test_gemm_mfma_m16n16k16_row_row_fp16fp16fp32():
    sch = tvm.tir.Schedule(gemm_mfma_m16n16k16_row_row_fp16fp16fp32)
    hip_mod = tvm.build(sch.mod, target="hip")
    print(hip_mod.imported_modules[0].get_source())
    A_np = np.random.uniform(-1, 1, [16, 16]).astype("float16")
    B_np = np.random.uniform(-1, 1, [16, 16]).astype("float16")
    C_np = np.zeros([16, 16]).astype("float32")

    ctx = tvm.rocm()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    hip_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("float16"), B_np.astype("float16"))

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mfma_m16n16k16_row_row_i8i8i32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 16], dtype="int8")
    B = T.match_buffer(b, [16, 16], dtype="int8")
    C = T.match_buffer(c, [16, 16], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 64)
    MultiA = T.decl_buffer([4], "int8", scope="local")
    MultiB = T.decl_buffer([4], "int8", scope="local")
    Accum = T.decl_buffer([4], "int32", scope="local")

    for i in range(4):
        Accum[i] = T.float64(0)
    for i in range(4):
        MultiA[i] = A[tx % 16, tx // 16 * 4 + i]
    for i in range(4):
        MultiB[i] = B[i + tx // 16 * 4, tx % 16]

    T.evaluate(T.tvm_mfma(
        "i32_16x16x16i8",
        "row",
        "row",
        "int32",
        "int32",
        "int32x4",
        MultiA.data,
        0,
        MultiB.data,
        0,
        Accum.data,
        0,
        dtype="int32x4",
    ))
    for mma_accum_c_id in range(4):
        C[tx // 16 * 4 + mma_accum_c_id, tx % 16] = Accum[mma_accum_c_id]


@tvm.testing.requires_rocm
def test_gemm_mfma_m16n16k16_row_row_i8i8i32():
    sch = tvm.tir.Schedule(gemm_mfma_m16n16k16_row_row_i8i8i32)
    hip_mod = tvm.build(sch.mod, target="hip")
    print(hip_mod.imported_modules[0].get_source())
    A_np = np.random.uniform(-1, 1, [16, 16]).astype("int8")
    B_np = np.random.uniform(-1, 1, [16, 16]).astype("int8")
    C_np = np.zeros([16, 16]).astype("int32")

    ctx = tvm.rocm()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    hip_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("int8"), B_np.astype("int8"))

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    # tvm.testing.main()
    # test_gemm_mfma_m16n16k4_row_row_fp32pf32fp32()
    test_gemm_mfma_m16n16k16_row_row_fp16fp16fp32()
    test_gemm_mfma_m16n16k16_row_row_i8i8i32()
