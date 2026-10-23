# RMSNorm benchmark: MLX mx.fast.rms_norm over the last axis, GB/s.
# Case list and shapes are the ones bench_rmsnorm_metal.nim runs.
# Methodology: one mx.eval per 20 norms, the median of 3 timed rounds.
#
# GB/s traffic model, the model bench_rmsnorm_metal.nim reports:
# - 2 source passes (sum of squares, scaled apply) plus 1 destination pass
# - 12 bytes per logical element for the unweighted kernel
# - the weight vector is rowSize floats total, cache resident, uncounted
#
# Byte accounting:
# - the timing counts logical elements, never the buffer behind the strided view
# - the buffer runs up to 2x the logical element count on the row-skip and stride-2 cases,
#   a buffer-cosize count would inflate those rows
#
# mx.fast.rms_norm normalizes over the last axis only, every case below
# normalizes the last axis.
#
# MLX exposes no unweighted fast path, the weight applies in every case.
# The weight stays cache resident, the traffic model is unchanged.
import math
import time
import mlx.core as mx

# (shape, strides, axis, iters) per rank, axis is always the last one
CASES2 = [
    ((4096, 4096), (4096, 1), 1, 50),   # row-major, dense rows
    ((4096, 4096), (1, 4096), 1, 50),   # column-major, rows 4096 elements apart
    ((4096, 4096), (8192, 1), 1, 50),   # row-skipped, rows 8192 elements apart
    ((64, 64), (64, 1), 1, 200),        # launch-bound small case, 64 x 64
    ((16384, 16384), (16384, 1), 1, 20),  # large row-major, 16384 squared
    ((4096, 4096), (8192, 2), 1, 20),   # gappy stride-2 rows
]
CASES2W = [
    ((4096, 4096), (4096, 1), 1, 50),   # weighted row-major, dense rows
]
CASES3 = [
    ((256, 128, 512), (65536, 512, 1), 2, 20),   # row-major, dense rows of 512
    ((256, 128, 512), (65536, 1, 128), 2, 20),   # last-axis stride 128, scattered within a row
    ((256, 128, 512), (512, 131072, 1), 2, 20),  # middle-stride slowest, dense rows of 512
    ((512, 512, 512), (262144, 512, 1), 2, 20),  # large row-major, 512 cubed
]
CASES4 = [
    ((64, 32, 64, 128), (8192, 524288, 128, 1), 3, 20),  # rank-4, dense rows of 128
]

def cosize(shape, strides):
    return 1 + sum((d - 1) * s for d, s in zip(shape, strides))

def rows(shape, axis):
    return math.prod(d for i, d in enumerate(shape) if i != axis)

def run(shape, strides, axis, iters):
    np = math.prod(shape)
    buf = mx.arange(cosize(shape, strides), dtype=mx.float32)
    a = mx.as_strided(buf, shape, strides)
    w = 1.0 + (mx.arange(shape[-1], dtype=mx.float32) % 5) * 0.25
    mx.eval(mx.fast.rms_norm(a, w, 1e-5))  # warmup
    batch = 20
    rounds = iters // batch
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(rounds):
            mx.eval([mx.fast.rms_norm(a, w, 1e-5) for _ in range(batch)])
        timings.append(time.perf_counter() - t0)
    dt = sorted(timings)[1]
    gbps = np * 12.0 * rounds * batch / dt / 1e9
    print(f"{np:>10}  {rows(shape, axis):>7}   {gbps:.1f}")

print("mx.fast.rms_norm last axis, Metal backend, batched (one mx.eval per 20 norms), GB/s = logical elements x 12 bytes")
print("        np      rows   GB/s")
for case in CASES2 + CASES2W + CASES3 + CASES4:
    run(*case)
