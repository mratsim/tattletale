# Axis-sum benchmark: MLX mx.sum per axis, GB/s. Mirrors bench_sum_axis_metal.nim.
# Cube (n, n, n), contiguous and last-two-swapped strides, batched methodology
# (one mx.eval per 20 sums). GB/s counts source bytes (n^3 * 4).
import time
import mlx.core as mx

CUBES = [64, 128, 256]

def iters_for(n):
    np = n * n * n
    return 200 if np <= 200_000 else (50 if np <= 1_000_000 else 20)

print("mx.sum axis, Metal backend, batched (one mx.eval per 20 sums)")
print("     n   layout   axis   GB/s")
for n in CUBES:
    iters = iters_for(n)
    for layout in ["contig", "swapped"]:
        a = mx.arange(n * n * n, dtype=mx.float32).reshape(n, n, n)
        if layout == "swapped":
            a = a.transpose(0, 2, 1)
        for axis in [0, 1, 2]:
            s = mx.sum(a, axis=axis)
            mx.eval(s)  # warmup
            assert s.size == n * n
            batch = 20
            rounds = iters // batch
            t0 = time.perf_counter()
            for _ in range(rounds):
                mx.eval([mx.sum(a, axis=axis) for _ in range(batch)])
            dt = time.perf_counter() - t0
            gbps = n * n * n * 4.0 * rounds * batch / dt / 1e9
            print(f"{n:>6}  {layout:>7}   {axis}    {gbps:.1f}")
