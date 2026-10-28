# SDPA benchmark: MLX mx.fast.scaled_dot_product_attention, effective TFLOP/s.
# Mirrors bench_scaled_dot_product_attention_metal.nim: same case list, same
# effective-flops model, GQA Hq=32 over Hkv=8, Dk=Dv=128.
# Dtype from argv: f32 (default) or bf16.
#
# Methodology: batched eval, one mx.eval per 20 calls, median of 3 timed
# rounds. The Nim side amortizes launch overhead with iters passes inside one
# kernel launch and reports the median of 3 launches; the batched eval gives
# the same amortization on the MLX side.
#
# Effective flops, the model the Nim bench reports:
# - only the key tiles a query tile visits count
# - kv tiles = ceil(S / NT), MT = NT = 32
# - causal: query tile q visits min((q*MT + MT - 1) / NT + 1, kvTiles) tiles
# - full: every query tile visits kvTiles
# - per visited tile: QK 2*MT*NT*Dk, PV 2*MT*NT*Dv, over Hq*B heads
#
# Causal alignment for L < S: the kernel contract is hi(q) = min(S, q + offset + 1)
# with offset 0, a top-left band. A query row touches keys 0 .. L-1 at most, so
# the MLX side slices k and v to the first min(S, L) keys on causal cases. The
# "causal" mask string is lower-right aligned, on the sliced square shapes the
# two alignments agree. The Nim kernel never reads the masked-out key tiles,
# the slice keeps both sides on identical math over identical key counts.
#
# Run from the worktree root:
#   uv run python workspace/ceramic/bench/sdpa_gpu/bench_scaled_dot_product_attention_mlx.py

import math
import sys
import time

import mlx.core as mx

DTYPE = mx.bfloat16 if (len(sys.argv) > 1 and sys.argv[1] == "bf16") else mx.float32

B, HQ, HKV, DK, DV = 1, 32, 8, 128, 128
MT, NT = 32, 32
BATCH = 20

# (L, S, causal, iters), the case list of bench_scaled_dot_product_attention_metal.nim
CASES = [
    (512, 512, True, 100),
    (1024, 1024, True, 40),
    (2048, 2048, True, 20),
    (2048, 2048, False, 20),
    (32, 2048, True, 100),
    (32, 8192, True, 100),
]


def make(shape):
    # logical index mod 7, times 0.25 plus 0.5, the Nim fill pattern
    n = math.prod(shape)
    return (mx.arange(n, dtype=mx.float32) % 7 * 0.25 + 0.5).reshape(shape).astype(DTYPE)


def causal_top_left_mask(l, s):
    return mx.arange(l)[:, None] >= mx.arange(s)[None, :]


def causal_bottom_right_mask(l, s):
    # the Nim kernel's band: offset = s - l, hi(q) = min(s, q + offset + 1)
    return mx.arange(l)[:, None] + (s - l) >= mx.arange(s)[None, :]


def ref_sdpa(q, k, v, scale, mask):
    # explicit math over GQA-expanded heads, the correctness anchor
    reps = q.shape[1] // k.shape[1]
    k = mx.repeat(k, reps, axis=1)
    v = mx.repeat(v, reps, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    scores = mx.where(mask, scores, mx.array(-mx.inf))
    return mx.softmax(scores, axis=-1) @ v


def check_math():
    # fused path vs explicit math, small causal shapes
    # square: "causal" string against a top-left mask
    q, k, v = make((1, 4, 16, 32)), make((1, 2, 16, 32)), make((1, 2, 16, 32))
    scale = 32 ** -0.5
    fused = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    want = ref_sdpa(q, k, v, scale, causal_top_left_mask(16, 16))
    err = mx.abs(fused - want).max().item()
    tol = 5e-2 if DTYPE == mx.bfloat16 else 1e-4
    assert err < tol, f"square causal mismatch, max err {err}"
    # L < S: the full-S lower-right op the benches time. This also verifies
    # the installed MLX aligns "causal" lower-right for L != S; an older or
    # different alignment fails loudly here instead of skewing parity numbers
    q, k, v = make((1, 4, 8, 32)), make((1, 2, 32, 32)), make((1, 2, 32, 32))
    fused = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    want = ref_sdpa(q, k, v, scale, causal_bottom_right_mask(8, 32))
    err = mx.abs(fused - want).max().item()
    assert err < tol, (
        f"L<S causal mismatch, max err {err} - the installed MLX 'causal' "
        "mask may not be lower-right aligned for L != S, Metal/MLX parity is invalid")


def visited_tiles(q_tiles, l, s, causal):
    # key tiles visited across the query-tile grid, the Nim kernel's band
    # (q_base = q_tile*MT + (s - l), bottom-right)
    kv_tiles = -(-s // NT)
    total = 0
    for q_tile in range(q_tiles):
        if causal:
            q_base = q_tile * MT + (s - l)
            total += max(0, min(kv_tiles, -(-(q_base + MT) // NT)))
        else:
            total += kv_tiles
    return total


def run_case(l, s, causal, iters):
    # full S keys, the same op the Metal bench times (bottom-right band)
    q = make((B, HQ, l, DK))
    k = make((B, HKV, s, DK))
    v = make((B, HKV, s, DV))
    scale = 1.0 / math.sqrt(DK)
    mask = "causal" if causal else None

    def call():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    mx.eval(call())  # warmup
    rounds = max(1, iters // BATCH)
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(rounds):
            mx.eval([call() for _ in range(BATCH)])
        timings.append(time.perf_counter() - t0)
    med = sorted(timings)[1] / (rounds * BATCH)
    flops = visited_tiles(-(-l // MT), l, s, causal) * HQ * B * 2 * MT * NT * (DK + DV)
    tflops = flops / med / 1e12
    print(f"L={l:>5} S={s:>5} causal={str(causal):>5}  median {med * 1e6:>10.1f} us  effective {tflops:>6.2f} TFLOP/s")


check_math()
print(f"SDPA MLX bench, B=1 Hq=32 Hkv=8 Dk=Dv=128 {'bf16' if DTYPE == mx.bfloat16 else 'f32'} (M4 Max), batched (one mx.eval per 20 calls)")
for case in CASES:
    run_case(*case)
