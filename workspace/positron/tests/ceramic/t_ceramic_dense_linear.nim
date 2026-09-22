## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_dense_linear.nim
##
## Ceramic dense linear suite, the kernel judged per element against the host reference
##
## - Out[m][n] = sum_k X[m][k] · W[n][k] over the row-major (N, K) weights
## X (M, K) → fp32 dot over Wᵀ row n → fp32 accumulator → one RNE store to Out (M, N)
##
## | subject     | contract                                                                                |
## | ----------- | --------------------------------------------------------------------------------------- |
## | naive side  | host fp32 dot products over the exact widenings, plus an fp64 cross-check               |
## | accumulator | kernel and naive side both accumulate fp32, one RNE to the storage element at the store |
## | regimes     | the same kernel body at M = 1 (GEMV) and M > 32 (tail M-tile)                           |
##
## | shape    | M  | N    | K    | TileC | family     | cases |
## | -------- | --- | ---- | ---- | ----- | ---------- | ----- |
## | gemv     | 1  | 128  | 64   | 64    | bf16, fp16 | 32    |
## | gemm     | 37 | 192  | 160  | 64    | bf16, fp16 | 16    |
## | out_proj | 1  | 4096 | 2048 | 64    | bf16       | 16    |
## | tilec32  | 1  | 32   | 2048 | 32    | bf16       | 16    |
## | multi    | 65 | 192  | 160  | 64    | bf16, fp16 | 8     |
##
## | note     | content                                                                                                                                                                                                             |
## | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | out_proj | the mega kernel's production binding, `dense_linear_tile_fwd[bfloat16, 2048, 4096, 64]` in the decode composition, a fresh monomorphization under the instantiation contract, so it carries a suite case of its own |
## | tilec32  | the mega kernel's stage-4 a/b decay and beta GEMV binding, `dense_linear_tile_fwd[bfloat16, 32, 2048, 32]`, a fresh monomorphization at the TileC = 32 boundary, so it carries a suite case of its own              |
## | multi    | the multi-M-tile regime (grid.y = 3, two consecutive full 32-row tiles plus a 1-row tail), the rowLimit composition across consecutive straddling tiles is the fragile path                                         |
## | N = 1    | the shared expert row GEMV cannot go through this kernel (N mod TileC), the router suite's shared-expert scalar entry covers it                                                                                     |
##
## Band model, stated before measurement, u32 = 2⁻²⁴ fp32, u_fam = 2⁻⁸ bf16 / 2⁻¹¹ fp16
##
## | bar        | bound                                          | covers                                  |
## | ---------- | ---------------------------------------------- | --------------------------------------- |
## | out (m, n) | 2·u_fam·abs(out) + 2·K·2⁻²⁴·Σk abs(x·w) + 2⁻²⁵ | the accumulator order, one RNE per side |
##
## - 2·K·2⁻²⁴·Σk abs(x·w) covers the accumulator order, both sides sit within
##   K·2⁻²⁴·Σ abs terms of the exact dot, the kernel's 16-wide mma chain vs
##   the naive sequential fp32 sum, differences within twice that bound
## - 2·u_fam·abs(out) covers the store round, the two sides round slightly different
##   fp32 accumulators, each RNE within u_fam of its operand
## - the 2⁻²⁵ floor covers the fp16 subnormal output grid, also the bf16 grid
##
## - the fp64 cross-check keeps the reference side audited, the naive fp32 dot
##   judged inside its own band of the exact dot
## - the measured divergence justifies the model, never sets the bar
## - adjudicated on Apple M4 Max with fresh seeded xorshift64 inputs

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/dense_linear
import ../naive/naive_rng
import ../naive/naive_tensors
import ceramic_pagebuf
import ceramic_fam

# ─── Device entries, one per (family dtype, shape) binding ────────────

const DenseLinearMsl = metal:
  proc cer_dense_linear_bf16_gemv(
      outp, x, w: ptr UncheckedArray[bfloat16],
      M, tx, ty: int32) {.global.} =
    dense_linear_tile_fwd[bfloat16, 128, 64, 64](outp, x, w, M, tx, ty)

  proc cer_dense_linear_f16_gemv(
      outp, x, w: ptr UncheckedArray[float16],
      M, tx, ty: int32) {.global.} =
    dense_linear_tile_fwd[float16, 128, 64, 64](outp, x, w, M, tx, ty)

  proc cer_dense_linear_bf16_gemm(
      outp, x, w: ptr UncheckedArray[bfloat16],
      M, tx, ty: int32) {.global.} =
    dense_linear_tile_fwd[bfloat16, 192, 160, 64](outp, x, w, M, tx, ty)

  proc cer_dense_linear_f16_gemm(
      outp, x, w: ptr UncheckedArray[float16],
      M, tx, ty: int32) {.global.} =
    dense_linear_tile_fwd[float16, 192, 160, 64](outp, x, w, M, tx, ty)

  proc cer_dense_linear_bf16_outproj(
      outp, x, w: ptr UncheckedArray[bfloat16],
      M, tx, ty: int32) {.global.} =
    dense_linear_tile_fwd[bfloat16, 4096, 2048, 64](outp, x, w, M, tx, ty)

  # the mega composition's stage-4 a/b decay and beta GEMV binding,
  # N mod TileC == 0 at the exact TileC = 32 boundary
  proc cer_dense_linear_bf16_tilec32(
      outp, x, w: ptr UncheckedArray[bfloat16],
      M, tx, ty: int32) {.global.} =
    dense_linear_tile_fwd[bfloat16, 32, 2048, 32](outp, x, w, M, tx, ty)

# ─── Host, the independent reference ─────────────────────────────────

const
  FloorSub = 2.9802322387695312e-8   # 2^-25, half the fp16 subnormal ulp,
                                     # the rounding floor at tiny outputs

proc naiveLinearF32(fam: Family, x, w: seq[uint16]; M, N, K: int): seq[float64] =
  ## Independent host reference at fp32 arithmetic over the exact widenings.
  ## This is the exact-dot form.
  ##
  ## - the bf16-rounded output form lives in `naiveDenseLinear` (naive_layer_ops)
  ## - both forms are judged against the kernel, each under its own band
  result = newSeq[float64](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += famWiden(fam, x[m * K + k]) * famWiden(fam, w[n * K + k])
      result[m * N + n] = acc.float64

proc naiveLinearF64(fam: Family, x, w: seq[uint16]; M, N, K: int): seq[float64] =
  ## Exact fp64 widening of the same dot, the naive side's own cross-check.
  result = newSeq[float64](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f64
      for k in 0 ..< K:
        acc += famWiden(fam, x[m * K + k]).float64 *
          famWiden(fam, w[n * K + k]).float64
      result[m * N + n] = acc

proc runCombo(engine: HwEngine; fam: Family, M, N, K, TileC, cases: int;
    seed: uint64; label: string; kernelName: string) =
  ## One (family dtype, shape) combination over `cases` independent seeded runs,
  ## judged per element under the band, case 0 relaunched bit-identical,
  ## `kernelName` selects the static binding
  let nOut = M * N
  let nX = M * K
  let nW = N * K
  var outB = allocPageBuf[uint16](nOut)
  var xB = allocPageBuf[uint16](nX)
  var wB = allocPageBuf[uint16](nW)
  defer:
    freePageBuf(outB); freePageBuf(xB); freePageBuf(wB)
  var outPA = outB.pa()
  var xPA = xB.pa()
  var wPA = wB.pa()
  let uFam = if fam == famBf16: 3.90625e-3 else: 4.8828125e-4
  let gridX = int32(N div TileC)
  let gridY = int32((M + 31) div 32)

  var worstUse = 0.0'f64
  var exact = 0
  var total = 0
  var launches = 0

  proc takeInputs(rng: var NaiveRng): tuple[x, w: seq[uint16]] =
    ## Seeded inputs, family-dtype bits for x and w.
    var xBits = newSeq[uint16](nX)
    var wBits = newSeq[uint16](nW)
    for i in 0 ..< nX:
      xBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
    for i in 0 ..< nW:
      wBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
    result = (xBits, wBits)

  proc load(bits: tuple[x, w: seq[uint16]]) =
    for i in 0 ..< nX:
      xB.hostPtr[i] = bits.x[i]
    for i in 0 ..< nW:
      wB.hostPtr[i] = bits.w[i]

  proc sentinels(bits: tuple[x, w: seq[uint16]]) =
    assertTailZero(outB, nOut)
    assertReadUnchanged(xB, bits.x)
    assertReadUnchanged(wB, bits.w)

  proc launch =
    for tx in 0 ..< gridX:
      for ty in 0 ..< gridY:
        engine.run << (grid: (1, 1, 1), blk: (32, 1, 1)) >>
          (kernelName, outPA, (xPA, wPA, int32(M), int32(tx), int32(ty)))
    inc launches, gridX * gridY

  proc snapOut(): seq[uint16] =
    result = newSeq[uint16](nOut)
    for i in 0 ..< nOut:
      result[i] = outB.hostPtr[i]

  var case0Snap: seq[uint16]
  var rng = initNaiveRng(seed)
  for caseId in 0 ..< cases:
    let bits = takeInputs(rng)
    let want = naiveLinearF32(fam, bits.x, bits.w, M, N, K)
    let want64 = naiveLinearF64(fam, bits.x, bits.w, M, N, K)
    load(bits)
    launch()
    sentinels(bits)
    for m in 0 ..< M:
      for n in 0 ..< N:
        let idx = m * N + n
        var sumAbs = 0.0'f64
        for k in 0 ..< K:
          sumAbs += abs(famWiden(fam, bits.x[m * K + k]).float64 *
            famWiden(fam, bits.w[n * K + k]).float64)
        let bar = 2.0 * uFam * abs(want[idx]) +
          2.0 * K.float64 * U32 * sumAbs + FloorSub
        let got = famWiden(fam, outB.hostPtr[idx]).float64
        let diff = abs(got - want[idx])
        doAssert diff <= bar,
          &"out outside the bar at (m {m}, n {n}, case {caseId}): " &
          &"{diff:.3e} > {bar:.3e}"
        worstUse = max(worstUse, diff / bar)
        if got == want[idx]:
          inc exact
        inc total
    # the naive side's own cross-check, the fp32 dot within its own band of fp64
    for m in 0 ..< M:
      for n in 0 ..< N:
        let idx = m * N + n
        var sumAbs = 0.0'f64
        for k in 0 ..< K:
          sumAbs += abs(famWiden(fam, bits.x[m * K + k]).float64 *
            famWiden(fam, bits.w[n * K + k]).float64)
        let barNaive = K.float64 * U32 * sumAbs
        doAssert abs(want[idx] - want64[idx]) <= barNaive,
          "naive fp32 dot outside its own band of fp64"
    if caseId == 0:
      case0Snap = snapOut()

  block determinism:
    var rng0 = initNaiveRng(seed)
    let bits0 = takeInputs(rng0)
    load(bits0)
    launch()
    sentinels(bits0)
    let again = snapOut()
    for i in 0 ..< nOut:
      doAssert again[i] == case0Snap[i], "out differs run to run"

  echo &"[{label} {famName(fam)}] cases={cases} launches={launches} " &
    &"worst bar usage {worstUse:.3f}, bit-exact {exact}/{total}"

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(DenseLinearMsl)
  runCombo(engine, famBf16, 1, 128, 64, 64, 32, 0xC04D0511'u64, "gemv",
    "cer_dense_linear_bf16_gemv")
  runCombo(engine, famF16, 1, 128, 64, 64, 32, 0xC04D0512'u64, "gemv",
    "cer_dense_linear_f16_gemv")
  runCombo(engine, famBf16, 37, 192, 160, 64, 16, 0xC04D0513'u64, "gemm tail",
    "cer_dense_linear_bf16_gemm")
  runCombo(engine, famF16, 37, 192, 160, 64, 16, 0xC04D0514'u64, "gemm tail",
    "cer_dense_linear_f16_gemm")
  runCombo(engine, famBf16, 1, 4096, 2048, 64, 16, 0xC04D0515'u64, "out_proj",
    "cer_dense_linear_bf16_outproj")
  runCombo(engine, famBf16, 1, 32, 2048, 32, 16, 0xC04D0516'u64, "tilec32",
    "cer_dense_linear_bf16_tilec32")
  runCombo(engine, famBf16, 65, 192, 160, 64, 8, 0xC04D0517'u64, "gemm multi-tile",
    "cer_dense_linear_bf16_gemm")
  runCombo(engine, famF16, 65, 192, 160, 64, 8, 0xC04D0518'u64, "gemm multi-tile",
    "cer_dense_linear_f16_gemm")
  echo "CERAMIC DENSE_LINEAR VERDICT: all cases inside the stated per-element bars"

main()
