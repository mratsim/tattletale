# Baseline benchmark: MLX mx.sum read bandwidth, GB/s. Mirrors bench_sum_metal.nim.
# Batched methodology: one mx.eval per 20 sums, the per-sum mx.eval sync
# inflated the dispatch overhead and understated the bandwidth.
import time
import mlx.core as mx

SIZES = [1_000, 16_000, 100_000, 200_000, 1_000_000, 4_000_000, 16_000_000]

def iters_for(n):
    return 200 if n <= 200_000 else (50 if n <= 1_000_000 else 20)

print("mx.sum, Metal backend, batched (one mx.eval per 20 sums)")
print("     N    iters   GB/s")
for n in SIZES:
    a = mx.arange(n, dtype=mx.float32) % 7 * 0.25
    iters = iters_for(n)
    batch = 20
    s = mx.sum(a)
    mx.eval(s)  # warmup
    assert float(s) > 0
    rounds = iters // batch
    t0 = time.perf_counter()
    for _ in range(rounds):
        mx.eval([mx.sum(a) for _ in range(batch)])
    dt = time.perf_counter() - t0
    gbps = n * 4.0 * rounds * batch / dt / 1e9
    print(f"{n:>9}  {iters:>6}   {gbps:.1f}")
