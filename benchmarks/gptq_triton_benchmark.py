import torch

import triton
import pytest

from vllm.model_executor.layers.quantization.gptq_quantized_matmul import gptq_qlinear

torch.manual_seed(42)

warmup = 5
rep = 10

@pytest.mark.parametrize("M, N, K, G",
    [ shape for shape in [(3072, 10240, 8192, 64), (3072, 8192, 28672, 224)]]
)
def test_benchmark(M, N, K, G):
    a = torch.rand((M, K), dtype=torch.float16, device="cuda:0")
    b = torch.randint(high=32**2-1, size=(K // 8, N), dtype=torch.int32, device="cuda:0")

    scales = torch.rand((G, N), dtype=torch.float16, device="cuda:0")
    qzeros = torch.randint(high=32**2-1, size=(G, N // 8), dtype=torch.int32, device="cuda:0")
    idx = torch.randint(high=qzeros.shape[0], size=(K,), dtype=torch.int32, device="cuda:0")

    d = torch.rand((M, N), dtype=torch.float32, device="cuda:0")

    res = triton.testing.do_bench(
        lambda: gptq_qlinear(a, b, d, scales, qzeros, idx, 4),
        warmup=warmup, rep=rep)

    print(f'SIZE: {M},{N},{K} RES: {res=}')


