import torch

import triton
import pytest

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gptq_quantized_matmul import gptq_qlinear


# export TRITON_CACHE_DIR=/root/.triton/cache
# first_asm_file=$(find $TRITON_CACHE_DIR -name "*.amdgcn" | head -n 1)

# TRITON_ALWAYS_COMPILE=1 TRITON_PRINT_AUTOTUNING=1 rocprofv2 -i /data/scripts/att.txt --plugin att $first_asm_file --mode file -d /data/profiling/gptq_triton/output python3 -m pytest -s /root/workspace/scripts/gptq_triton_benchmark.py

# /data/scripts/att.txt
# att: TARGET_CU=0
# SE_MASK=0x0
# SIMD_SELECT=0xF
# ISA_CAPTURE_MODE=2
# KERNEL=_quantized_matmul

# TRITON_ALWAYS_COMPILE=1 TRITON_PRINT_AUTOTUNING=1 python3 -m pytest -s /root/workspace/scripts/gptq_triton_benchmark.py

torch.manual_seed(42)

warmup = 5
rep = 10

def gptq_gemm(a, b, d, scales, qzeros, idx):
    output = ops.gptq_gemm(a, b, scales, qzeros, idx, False, 4)
    if d is not None:
        output.add_(d)
    return output

@pytest.mark.parametrize("M, N, K, G",
   #[ shape for shape in [(3072, 10240, 8192, 64), (3072, 8192, 28672, 224)]]
    [ shape for shape in [(3072, 10240, 8192, 64)]]
)
def test_benchmark(M, N, K, G):
    a = torch.rand((M, K), dtype=torch.float16, device="cuda:0")
    b = torch.randint(high=32**2-1, size=(K // 8, N), dtype=torch.int32, device="cuda:0")

    scales = torch.rand((G, N), dtype=torch.float16, device="cuda:0")
    qzeros = torch.randint(high=32**2-1, size=(G, N // 8), dtype=torch.int32, device="cuda:0")
    idx = torch.randint(high=qzeros.shape[0], size=(K,), dtype=torch.int32, device="cuda:0")

    d = torch.rand((M, N), dtype=torch.float16, device="cuda:0")

    res = triton.testing.do_bench(
        lambda: gptq_qlinear(a, b, d, scales, qzeros, idx, 4),
        warmup=warmup, rep=rep)
    print(f'SIZE: {M},{N},{K} Triton RES: {res=}')

    res = triton.testing.do_bench(
        lambda: gptq_gemm(a, b, d, scales, qzeros, idx),
        warmup=warmup, rep=rep)
    print(f'SIZE: {M},{N},{K} Native RES: {res=}')

    triton_output = gptq_qlinear(a, b, d, scales, qzeros, idx, 4)
    native_output = gptq_gemm(a, b, d, scales, qzeros, idx)

    if not torch.equal(triton_output, native_output):
        print("Tensors are not equal")
        print(f"{triton_output=}")
        print(f"{native_output=}")
