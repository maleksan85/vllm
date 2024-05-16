import triton
import triton.language as tl
import torch 


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=0,
            num_warps=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit()
def _h100_quantized_matmul(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, idx_ptr,
                             stride_am, stride_ak,
                             stride_bk, stride_bn,
                             stride_cm, stride_cn,
                             stride_scales,
                             stride_zeros,
                             M, N, K,
                             BLOCK_SIZE_M: tl.constexpr, 
                             BLOCK_SIZE_N: tl.constexpr, 
                             BLOCK_SIZE_K: tl.constexpr,
                             GROUP_SIZE_M: tl.constexpr,
):
    bits = 4
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

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                order=(1,0))  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    
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

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # (BLOCK_SIZE_M, BLOCK_SIZE_N)
    for k in range(0, total_blocks_k):
        g_idx = tl.load(idx_ptrs)

        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        # scales = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.int32) + 1
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # zeros = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.int32) + 1
        zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros).to(tl.uint32)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        zeros = (zeros >> zeros_shifter[None, :]) & 0xF # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        zeros = zeros + 1 # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        a = tl.load(a_block_ptr, boundary_check=(0,1))  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs).to(tl.uint32) # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        b = ((b >> shifter[:, None]) & 0xF).to(tl.float16) # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b = (b - zeros) * scales

        acc += tl.dot(a,b) # (BLOCK_SIZE_M, BLOCK_SIZE_N)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        idx_ptrs += BLOCK_SIZE_K

    acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class h100_qlinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, scales, zeros, idx):
        c = torch.zeros((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)

        # print("========================================================================")
        # print(f"{a.shape=},{a.dtype=}")
        # print(f"{b.shape=},{b.dtype=}")
        # print(f"{c.shape=},{c.dtype=}")
        # print(f"{scales.shape=},{scales.dtype=}")
        # print(f"{zeros.shape=},{zeros.dtype=}")
        # print(f"{idx.shape=},{idx.dtype=}")

        grid = lambda META: (  # noqa: E731
            triton.cdiv(c.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(c.shape[1], META["BLOCK_SIZE_N"]),
        )

        print(f"idx values out of 63 {idx[idx > 63]=}")

        k = _h100_quantized_matmul[grid](
            a, b, c, scales, zeros, idx,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0),
            zeros.stride(0),
            a.shape[0], b.shape[1], a.shape[1]
        )

        # print(f"{total_blocks_m=} x {total_blocks_n=} = {total_programs=}")
        return c
        

h100_qlinear = h100_qlinear.apply