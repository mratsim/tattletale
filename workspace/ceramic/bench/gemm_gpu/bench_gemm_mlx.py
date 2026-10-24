# Baseline benchmark: MLX mx.matmul on the CUDA backend, TFLOP/s.
# Companion of bench_gemm_cuda.nim (Tattletale Ceramic tile-GEMM, f16 in,
# f32 accumulate, one launch looping the GEMM `iters` times).
#
# Methodology, mirroring bench_sum_mlx.py:
# - one mx.eval per batch of matmuls, the per-matmul mx.eval sync inflates
#   the dispatch overhead and understates the throughput
# - f16 inputs, f16 output (MLX default), the flop count is dtype-independent
# - warmup before timing, the sizes and iteration counts mirror
#   bench_gemm_cuda.nim: >=4096: 20, >=2048: 100, else 1000
import time
import mlx.core as mx

SIZES = [256, 512, 1024, 2048, 4096]

def iters_for(n):
    return 20 if n >= 4096 else (100 if n >= 2048 else 1000)

print("mx.matmul, CUDA backend, batched (one mx.eval per batch)")
print("     MNK   iters   TFLOP/s")
for mnk in SIZES:
    m = n = k = mnk
    iters = iters_for(mnk)
    # deterministic fill, values don't matter for throughput; f16 arange
    # keeps the peak footprint down, this box has under 1 GiB free VRAM
    a = (mx.arange(m * k, dtype=mx.float16) % 13).reshape(m, k)
    b = (mx.arange(n * k, dtype=mx.float16) % 17).reshape(n, k)
    batch = 20 if mnk <= 1024 else 1
    d = a @ b
    mx.eval(d)  # warmup
    assert d.shape == (m, n)
    del a, b, d
    mx.clear_cache()
    # rebuild after the cache trim, the timed section re-fills a and b
    a = (mx.arange(m * k, dtype=mx.float16) % 13).reshape(m, k)
    b = (mx.arange(n * k, dtype=mx.float16) % 17).reshape(n, k)
    rounds = max(1, iters // batch)
    t0 = time.perf_counter()
    for _ in range(rounds):
        mx.eval([a @ b for _ in range(batch)])
    dt = time.perf_counter() - t0
    tflops = 2.0 * m * n * k * rounds * batch / dt / 1e12
    print(f"{mnk:>9}  {iters:>6}   {tflops:>8.2f}")
    del a, b
    mx.clear_cache()
