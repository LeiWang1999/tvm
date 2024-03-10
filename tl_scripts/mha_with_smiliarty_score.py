import torch
from tvm import tl
import tvm
import tvm.tl.language as T
from functools import partial
from einops import rearrange, repeat
import math


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), scores


def pytorch_mha(q, k, v, causal=True):
    return attention_ref(q, k, v, causal=causal)


def flashattn(batch, heads, seq_len, dim, is_casual, block_M, block_N):
    batch = tvm.te.var("batch")
    heads = tvm.te.var("heads")
    # scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    scale = (1.0 / dim) ** 0.5
    shape = [T.int32(batch), T.int32(seq_len), T.int32(heads), T.int32(dim)]
    score_shape = [
        T.int32(batch),
        T.int32(heads),
        T.int32(seq_len),
        T.int32(seq_len),
    ]  # Shape for similarity scores
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
        Q: T.Buffer(shape, dtype),
        K: T.Buffer(shape, dtype),
        V: T.Buffer(shape, dtype),
        Output: T.Buffer(shape, dtype),
        Similarity_Scores: T.Buffer(score_shape, accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(T.int32(seq_len), T.int32(block_M)),
            T.int32(heads),
            T.int32(batch),
            threads=T.int32(128),
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([T.int32(block_M), T.int32(dim)], dtype)
            Q_local = T.alloc_fragment([T.int32(block_M), T.int32(dim)], dtype)
            K_shared = T.alloc_shared([T.int32(block_N), T.int32(dim)], dtype)
            V_shared = T.alloc_shared([T.int32(block_N), T.int32(dim)], dtype)
            acc_s = T.alloc_fragment([T.int32(block_M), T.int32(block_N)], accum_dtype)
            acc_s_cast = T.alloc_fragment([T.int32(block_M), T.int32(block_N)], dtype)
            acc_o = T.alloc_fragment([T.int32(block_M), T.int32(dim)], accum_dtype)
            scores_max = T.alloc_fragment([T.int32(block_M)], accum_dtype)
            scores_max_prev = T.alloc_fragment([T.int32(block_M)], accum_dtype)
            scores_scale = T.alloc_fragment([T.int32(block_M)], accum_dtype)
            scores_sum = T.alloc_fragment([T.int32(block_M)], accum_dtype)
            logsum = T.alloc_fragment([T.int32(block_M)], accum_dtype)

            T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
            T.copy(
                Q[
                    T.int32(bz),
                    T.int32(bx) * T.int32(block_M) : (T.int32(bx) + T.int32(1)) * T.int32(block_M),
                    T.int32(by),
                    T.int32(0) : T.int32(dim),
                ],
                Q_shared,
            )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.copy(Q_shared, Q_local)
            for i, j in T.Parallel(T.int32(block_M), T.int32(dim)):
                Q_local[i, j] *= scale
            loop_range = (
                T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
            )
            for k in T.Pipelined(T.int32(loop_range), num_stages=1):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                if is_casual:
                    for i, j in T.Parallel(T.int32(block_M), T.int32(block_N)):
                        acc_s[i, j] = T.if_then_else(
                            bx * T.int32(block_M) + i >= k * T.int32(block_N) + j,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                else:
                    T.clear(acc_s)
                T.gemm(Q_local, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * T.int32(block_N) : (k + 1) * T.int32(block_N), by, :], V_shared)
                T.copy(scores_max, scores_max_prev)
                T.copy(
                    acc_s,
                    Similarity_Scores[
                        bz, by, bx * block_M : (bx + 1) * block_M, k * block_N : (k + 1) * block_N
                    ],
                )
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(T.int32(block_M)):
                    scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(T.int32(block_M), T.int32(dim)):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(T.int32(block_M), T.int32(block_N)):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(T.int32(block_M)):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(T.int32(block_M), T.int32(dim)):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, bx * T.int32(block_M) : (bx + 1) * T.int32(block_M), by, :])

    return main


if __name__ == "__main__":
    BATCH, H, N_CTX, D_HEAD = 16, 12, 1024, 256
    casual = True
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if casual:
        total_flops *= 0.5
    BLOCK_M = 64
    BLOCK_N = 64 if D_HEAD <= 128 else 32
    program = flashattn(BATCH, H, N_CTX, D_HEAD, casual, BLOCK_M, BLOCK_N)
    ref_program = partial(pytorch_mha, casual=casual)
    mod, params = tl.lower(program)
    print(mod.imported_modules[0].get_source())
    mod = tl.Profiler(mod, params, [3, 4], tl.TensorSupplyType.Normal, opt_shapes={
        "batch": BATCH, 
        "heads": H, 
    })
    mod.assert_allclose(pytorch_mha, rtol=0.01, atol=0.01)
    print("Pytorch MHA:")
    latency = mod.do_bench(pytorch_mha)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    print("Ours:")
    latency = mod.do_bench(mod)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
