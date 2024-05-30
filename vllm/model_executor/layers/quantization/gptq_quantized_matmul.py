import triton
import triton.language as tl
import torch 



# def autotune_config():
#     return [
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": bsm,
#                 "BLOCK_SIZE_N": bns,
#                 "BLOCK_SIZE_K": bnk,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=0,
#             num_warps=ws,
#         )
#         for bsm in [16, 32, 64, 128, 256]
#         for bns in [16, 32, 64, 128, 256]
#         for bnk in [16, 32, 64, 128, 256]
#         for ws in [2, 4, 8, 16, 32]
#     ]

# @triton.autotune(
#     configs=autotune_config(),
#     key=['M', 'N', 'K'],
# )
@triton.autotune(
    configs=[
        # triton.Config(
        #     {
        #         "BLOCK_SIZE_M": 128,
        #         "BLOCK_SIZE_N": 64,
        #         "BLOCK_SIZE_K": 64,
        #         "GROUP_SIZE_M": 4,
        #         "waves_per_eu": 6
        #     },
        #     num_stages=0,
        #     num_warps=8,
        # ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 2,
                "waves_per_eu": 6
            },
            num_stages=0,
            num_warps=8,
        )
    ],
    key=["M", "N", "K"],
)
@triton.jit()
def _quantized_matmul(a_ptr, b_ptr, c_ptr, d_ptr, scales_ptr, zeros_ptr, idx_ptr,
                        stride_am, stride_ak,
                        stride_bk, stride_bn,
                        stride_cm, stride_cn,
                        stride_scales,
                        stride_zeros,
                        M, N, K,
                        bits,
                        maxq,
                        BLOCK_SIZE_M: tl.constexpr, 
                        BLOCK_SIZE_N: tl.constexpr, 
                        BLOCK_SIZE_K: tl.constexpr,
                        GROUP_SIZE_M: tl.constexpr,
):
    infearure_per_bits = 32 // bits
    
    pid = tl.program_id(axis=0)

    total_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    total_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_blocks_k = tl.cdiv(K, BLOCK_SIZE_K)

    num_blocks_in_group = GROUP_SIZE_M * total_blocks_n
    group_id = pid // num_blocks_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(total_blocks_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_blocks_in_group) // (group_size)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = offs_am[:, None] < M
    
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + (
        (offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)

    idx_ptrs = idx_ptr + offs_k # (BLOCK_SIZE_K)
    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + offs_bn # (BLOCK_SIZE_N)
    zeros_ptrs = zeros_ptr + (offs_bn // infearure_per_bits) # (BLOCK_SIZE_N)

    shifter = (offs_k % infearure_per_bits) * bits # (BLOCK_SIZE_K)
    zeros_shifter = (offs_bn % infearure_per_bits) * bits # (BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.dtype.element_ty) # (BLOCK_SIZE_M, BLOCK_SIZE_N)
    if (d_ptr):
        acc = tl.load(d_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None] + tl.arange(0, BLOCK_SIZE_N))

    for k in range(0, total_blocks_k):
        g_idx = tl.load(idx_ptrs)

        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros) # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        zeros = zeros >> zeros_shifter[None, :] # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        zeros = zeros & maxq # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        zeros = zeros + 1 # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        b = tl.load(b_ptrs) # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
        b = b >> shifter[:, None]
        b = b & maxq
        b = b.to(tl.float16) # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        b = b - zeros
        b = b * scales

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        acc += tl.dot(a, b, out_dtype=c_ptr.dtype.element_ty) # (BLOCK_SIZE_M, BLOCK_SIZE_N)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        idx_ptrs += BLOCK_SIZE_K

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class gptq_qlinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, d, scales, zeros, idx, bits):
        c = torch.zeros((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)

        grid = lambda META: (  # noqa: E731
            triton.cdiv(c.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(c.shape[1], META["BLOCK_SIZE_N"]),
        )

        _quantized_matmul[grid](
            a, b, c, d, scales, zeros, idx,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0),
            zeros.stride(0),
            a.shape[0], b.shape[1], a.shape[1],
            bits=bits, maxq=2**bits-1
        )

        return c
        
gptq_qlinear = gptq_qlinear.apply