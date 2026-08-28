## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/tests/manual_tile_attn_fp16 \
##   --nimcache:nimcache/tests/manual_tile_attn_fp16 \
##   workspace/ceramic/tests/kernels_tiles/manual_tile_attn_fp16.nim

import std/[strformat, strutils]
import workspace/crucible
import workspace/libtorch
import ../tile_test_utils
import ../../src/kernels/k_tile_attn

# ═════════════════════════════════════════════════════════════════════════
#  The thin {.global.} launchers, one per head dim (D = 64, 128).
#  First param = the output buffer o: engine.run binds the separate outBuf argument to it and the args tuple to the rest.
# ═════════════════════════════════════════════════════════════════════════

const attnMsl = metal:
  proc attnD64(o: ptr UncheckedArray[float32],
               q, k, v: ptr UncheckedArray[float16], H, N: int32) {.global.} =
    attn_fwd(q, k, v, o, H, N, 64)
  proc attnD128(o: ptr UncheckedArray[float32],
                q, k, v: ptr UncheckedArray[float16], H, N: int32) {.global.} =
    attn_fwd(q, k, v, o, H, N, 128)

# ═════════════════════════════════════════════════════════════════════════
#  Reference: f16-exact Q/K/V + torch SDPA (libtorch from the .venv)
# ═════════════════════════════════════════════════════════════════════════

func f16Exact(b, h, n, d, seed: int): float32 =
  ## Deterministic fp16-exact value in [-2, 2). Every value is an fp16 grid point, so fp32ToFp16 is exact.
  float32((seed * b + 7 * h + 11 * n + 13 * d) mod 32) / 8.0'f32 - 2.0'f32

proc torchSdpa(B, H, N, D, seedQ, seedK, seedV: int):
              tuple[q, k, v: seq[uint16]; o: seq[float32]] =
  ## fp16-exact Q/K/V (u16 bits for the kernel) + the reference
  ## O = SDPA(Q, K, V) computed by torch (libtorch from the .venv) in fp32, the dumb reference.
  let n = B * H * N * D
  var qf = newSeq[float32](n)
  var kf = newSeq[float32](n)
  var vf = newSeq[float32](n)
  var i = 0
  for b in 0 ..< B:
    for h in 0 ..< H:
      for s in 0 ..< N:
        for d in 0 ..< D:
          qf[i] = f16Exact(b, h, s, d, seedQ)
          kf[i] = f16Exact(b, h, s, d, seedK)
          vf[i] = f16Exact(b, h, s, d, seedV)
          inc i
  result.q = newSeq[uint16](n)
  result.k = newSeq[uint16](n)
  result.v = newSeq[uint16](n)
  for i in 0 ..< n:
    result.q[i] = fp32ToFp16(qf[i])
    result.k[i] = fp32ToFp16(kf[i])
    result.v[i] = fp32ToFp16(vf[i])
  let q = toTensor(qf).reshape(B, H, N, D)
  let k = toTensor(kf).reshape(B, H, N, D)
  let v = toTensor(vf).reshape(B, H, N, D)
  let o = scaled_dot_product_attention(q, k, v)
  result.o = newSeq[float32](n)
  let p = o.data_ptr(float32)
  for i in 0 ..< n:
    result.o[i] = p[i]

# ═════════════════════════════════════════════════════════════════════════
#  Test
# ═════════════════════════════════════════════════════════════════════════

proc runAttnKernel(engine: var auto; kernel: string; B, H, N: int;
                   Qh, Kh, Vh: seq[uint16]; outO: var seq[float32]) =
  ## Runs one attention kernel for the shape: grid (N/8, H, B) × blk 32.
  ## engine.run binds the separate output buffer (outO) to the launcher's first param
  ## (o) and the args tuple to the rest: (Qh, Kh, Vh) → (q, k, v), then H, N.
  # Guard: the KV loop steps 8-row blocks, so N must be a multiple of 8
  # (a partial KV block would corrupt the online softmax).
  doAssert N mod 8 == 0
  let qBlocks = N div 8
  engine.run << (grid: (qBlocks, H, B), blk: (32, 1)) >> (
    kernel, outO, (Qh, Kh, Vh, int32(H), int32(N)))

proc checkAttn(engine: var auto; kernel: string; B, H, N, D: int;
               data: tuple[q, k, v: seq[uint16]; o: seq[float32]]) =
  ## Runs the kernel for the shape and compares against torch SDPA (fp32,
  ## libtorch from the .venv) with the 1e-2 tolerance.
  let n = B * H * N * D
  let qBlocks = N div 8
  var outF = newSeq[float32](n)
  runAttnKernel(engine, kernel, B, H, N, data.q, data.k, data.v, outF)

  var ok = true
  var worst = 0.0'f32
  var firstBad = ""
  var nBad = 0
  for i in 0 ..< n:
    let d = abs(outF[i] - data.o[i])
    if d > worst: worst = d
    if d != d or d > 1e-2'f32:      # NaN is a failure, not a pass
      ok = false
      inc nBad
      if firstBad.len == 0:
        let idx = i div D
        firstBad = &"(b={idx div (H*N)}, h={(idx div N) mod H}, " &
                   &"n={idx mod N}, d={i mod D}): got {outF[i]}, want {data.o[i]}"
  echo &"SDPA B={B} H={H} N={N} D={D} ({kernel}, " &
       &"grid ({qBlocks},{H},{B}), KV loop {N div 8} iters): ",
       (if ok: &"PASS — worst |Δ| = {worst} (tolerance 1e-2)"
        else: &"FAIL — {nBad} bad, e.g. {firstBad}, worst |Δ| = {worst}")
  if not ok:
    quit 1

proc runTests() =   # engines are RAII, so keep them function-local
  checkCasePins()          # the hash cases must not drift
  var engine = bkMetal.init()
  engine.ingest(attnMsl)
  echo attnMsl            # keep the generated MSL inspectable

  var cases = initShapeCases("attn")
  for caseIdx in 0 ..< 3:
    let b = cases.nextBytes()
    let B = caseInRange(b, 0, 1, 2)
    let H = caseInRange(b, 1, 1, 2)
    let N = 8 * caseInRange(b, 4, 1, 12)   # N ∈ {8..96}, multiples of 8
    let g = torchSdpa(B, H, N, D = 64, seedQ = 1, seedK = 4, seedV = 8)
    checkAttn(engine, "attnD64", B, H, N, 64, g)
    let g128 = torchSdpa(B, H, N, D = 128, seedQ = 11, seedK = 14, seedV = 18)
    checkAttn(engine, "attnD128", B, H, N, 128, g128)

when isMainModule:
  runTests()
