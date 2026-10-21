# Axis-sum benchmark: MLX mx.sum per axis, GB/s.
# One mx.eval per 20 sums. GB/s counts logical source bytes.
# The cases match bench_sum_axis_metal.nim.
import math
import time
import mlx.core as mx

# (shape, strides, axis, iters) per rank
CASES2 = [
    ((4096, 4096), (4096, 1), 1, 50),
    ((4096, 4096), (1, 4096), 1, 50),
    ((4096, 4096), (8192, 1), 0, 50),
    ((64, 64), (64, 1), 1, 200),
    ((16384, 16384), (16384, 1), 1, 20),
    ((4096, 4096), (8192, 2), 1, 20),
]
CASES3 = [
    ((256, 128, 512), (65536, 512, 1), 0, 20),
    ((256, 128, 512), (65536, 1, 128), 1, 20),
    ((256, 128, 512), (512, 131072, 1), 2, 20),
    ((512, 512, 512), (262144, 512, 1), 0, 20),
    ((256, 128, 512), (131072, 1024, 1), 0, 20),
]
CASES4 = [
    ((64, 32, 64, 128), (8192, 524288, 128, 1), 1, 20),
]

def cosize(shape, strides):
    return 1 + sum((d - 1) * s for d, s in zip(shape, strides))

def run(shape, strides, axis, iters):
    buf = mx.arange(cosize(shape, strides), dtype=mx.float32)
    a = mx.as_strided(buf, shape, strides)
    s = mx.sum(a, axis=axis)
    mx.eval(s)  # warmup
    assert s.size == math.prod(d for i, d in enumerate(shape) if i != axis)
    batch = 20
    rounds = iters // batch
    t0 = time.perf_counter()
    for _ in range(rounds):
        mx.eval([mx.sum(a, axis=axis) for _ in range(batch)])
    dt = time.perf_counter() - t0
    gbps = math.prod(shape) * 4.0 * rounds * batch / dt / 1e9
    print(f"{str(shape):>22} {str(strides):>20}   {axis}    {gbps:.1f}")

print("mx.sum axis, Metal backend, batched (one mx.eval per 20 sums)")
print("                 shape              strides   axis   GB/s")
for case in CASES2 + CASES3 + CASES4:
    run(*case)
