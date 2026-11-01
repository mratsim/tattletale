# Baseline benchmark: PyTorch torch.mm, bf16, TFLOP/s.
# Companion of bench_gemm_smem_cuda.nim, the Tattletale Ceramic smem
# tile-GEMM bench:
# - bf16 in, f32 accumulate
# - one launch looping the GEMM `iters` times
#
# Methodology, mirroring bench_gemm_mlx.py:
# - timing over back-to-back kernel launches, one sync per timed block,
#   a per-matmul sync measures launch overhead instead of the GPU
# - bf16 inputs, bf16 output (torch.mm default), the flop count is
#   dtype-independent
# - warmup before timing, the sizes and iteration counts mirror
#   bench_gemm_smem_cuda.nim: >=4096: 20, >=2048: 100, else 1000
import time
import torch

SIZES = [256, 512, 1024, 2048, 4096]

def iters_for(n):
    return 20 if n >= 4096 else (100 if n >= 2048 else 1000)

assert torch.cuda.is_available(), "bench_gemm_torch needs a CUDA device"
print("torch.mm, bf16, CUDA, batched (one sync per timed block)")
print("     MNK   iters   TFLOP/s")
for mnk in SIZES:
    m = n = k = mnk
    iters = iters_for(mnk)
    # deterministic fill, values don't matter for throughput
    # int32 arange casts to bf16 once, a bf16 arange loses precision
    a = (torch.arange(m * k, device="cuda", dtype=torch.int32) % 13).to(torch.bfloat16).reshape(m, k)
    b = (torch.arange(n * k, device="cuda", dtype=torch.int32) % 17).to(torch.bfloat16).reshape(n, k)
    batch = 20 if mnk <= 1024 else 1
    rounds = max(1, iters // batch)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        torch.mm(a, b, out=d)
    torch.cuda.synchronize()  # warmup
    assert d.shape == (m, n)
    t0 = time.perf_counter()
    for _ in range(rounds):
        for _ in range(batch):
            torch.mm(a, b, out=d)
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = 2.0 * m * n * k * rounds * batch / dt / 1e12
    print(f"{mnk:>9}  {iters:>6}   {tflops:>8.2f}")
    del a, b, d
    torch.cuda.empty_cache()
