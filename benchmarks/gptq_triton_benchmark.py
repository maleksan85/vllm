import torch

import triton
import pytest

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gptq_quantized_matmul import gptq_qlinear, dequant_weight


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


def python_dequant(a, b, d, qzeros, scales, idx, bits = 4):
    _broadcast = torch.arange(0, b.shape[0] * 8, device=b.device, dtype=b.dtype) // 8

    _shift = (torch.arange(0, b.shape[0] * 8, device=b.device, dtype=b.dtype) % 8) * bits
    _shift = _shift[:, None]

    output = b.index_select(0, _broadcast)
    output = (output >> _shift) & (2**bits-1)
    output = output.to(dtype=a.dtype)

    _scales = scales.index_select(0, idx)

    _qzeros = qzeros.index_select(0, idx)
    _broadcast = torch.zeros((_qzeros.shape[0], _qzeros.shape[1], 32 // bits), device=qzeros.device, dtype=qzeros.dtype)
    _qzeros = _qzeros.unsqueeze_(-1) + _broadcast
    _shift = torch.arange(0, 8, device=b.device, dtype=b.dtype) * bits
    _qzeros = (_qzeros >> _shift) & (2**bits-1)
    _qzeros = _qzeros.reshape(b.shape[0] * 8, b.shape[1]).to(dtype=a.dtype)

    output = (output - _qzeros - 1) * _scales

    return output


def python_gptq(a, b, d, qzeros, scales, idx, bits = 4):
    output = python_dequant(a, b, d, qzeros, scales, idx, bits)
    return torch.matmul(a, output)

def gptq_gemm(a, b, d, qzeros, scales, idx):
    output = ops.gptq_gemm(a, b, qzeros, scales, idx, False, 4)
    if d is not None:
        output.add_(d)
    return output

@pytest.mark.parametrize("M, N, K, G",
    # [ shape for shape in [(3072, 10240, 8192, 64), (3072, 8192, 28672, 224)]]
    [ shape for shape in [(3072, 10240, 8192, 64)]]
)
def test_benchmark(M, N, K, G):
    a = torch.rand((M, K), dtype=torch.float16, device="cuda:0")
    b = torch.randint(high=2**31, size=(K // 8, N), dtype=torch.int32, device="cuda:0")

    scales = torch.rand((G, N), dtype=torch.float16, device="cuda:0")
    qzeros = torch.randint(high=2**31, size=(G, N // 8), dtype=torch.int32, device="cuda:0")
    idx = torch.randint(high=qzeros.shape[0], size=(K,), dtype=torch.int32, device="cuda:0")

    d = torch.rand((M, N), dtype=torch.float16, device="cuda:0")

    res = triton.testing.do_bench(
        lambda: gptq_qlinear(a, b, d, qzeros, scales, idx, 4),
        warmup=warmup, rep=rep)

    print("")
    print(f'SIZE: {M},{N},{K} Triton RES: {res=}')

    res = triton.testing.do_bench(
        lambda: gptq_gemm(a, b, d, qzeros, scales, idx),
        warmup=warmup, rep=rep)
    print(f'SIZE: {M},{N},{K} Native RES: {res=}')
    
    # py_dequant = python_dequant(a, b, d, qzeros, scales, idx, 4)
    # tr_dequant = dequant_weight(a, b, qzeros, scales, idx, 4)
    # print("")
    # print(f"{torch.equal(tr_dequant, py_dequant)=} error is: \
    #         {torch.mean(torch.abs(tr_dequant - py_dequant)) / torch.mean(torch.abs(py_dequant))}")


    triton_output = gptq_qlinear(a, b, d, qzeros, scales, idx, 4)
    python_output = python_gptq(a, b, d, qzeros, scales, idx)
    native_output = gptq_gemm(a, b, d, qzeros, scales, idx)

    print("")
    print(f"{torch.equal(python_output, native_output)=} error is: \
            {torch.mean(torch.abs(python_output - native_output)) / torch.mean(torch.abs(python_output))}")
    
    print(f"{torch.equal(triton_output, native_output)=} error is: \
            {torch.mean(torch.abs(triton_output - native_output)) / torch.mean(torch.abs(native_output))}")

    print(f"{torch.equal(triton_output, python_output)=} error is: \
            {torch.mean(torch.abs(triton_output - python_output)) / torch.mean(torch.abs(python_output))}")

