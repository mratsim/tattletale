# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 trellis-dequant GEMM fixture replay (Metal).
##
## Stage 0 checks the decode contract host-side:
##
## - the value index is a bijection inverting `trellisValueCoords`, agreeing
##   with the v1 production decode's word formula
## - pack.cu-format fixture streams roundtrip through the dq funnel window read,
##   recovering the packed code in the window's low `bits` bits
##
## Stage 1 replays the trellis dequant bit-exact against the CPU reference,
## every bits level with all three codebooks.
##
## Stage 2 replays the GEMM banded against the float64 reference, one case per
## step of this progression:
##
## - one 16-position chunk at bits=2, then bits=4 and n=256 tiles
## - multi-chunk m (37 positions over three in-kernel m chunks)
## - the mcg and mul1 codebooks, fp16 and fp32 output paths, suh on or off
## - undersized grids: grid.x = 1 grid-strides every output tile, grid.y = 1
##   walks all three m chunks, then both axes narrow at once
##
## Bands, rationale in the template header:
##
## - dequant: bit-exact
## - GEMM fp32 out: rtol 1e-3 + abstol 1e-3 over the float64 reference
## - GEMM fp16 out: rtol 2e-3 + abstol 5e-2, one fp16 ulp of |C| beyond the fp32
##   accumulation gap
##
## Run from the worktree root:
##   nim c -r --threads:off -o:build/tests/t_exl3_gemm_metal \
##     workspace/positron/tests/exl3_gemm/t_exl3_gemm_metal.nim

import std/[random, strformat]
import workspace/crucible
import workspace/ceramic/src/macros/static_for
import ./exl3_gemm_template

# v1 production decode's word formula (quant_exl3_ops.nim dequantTrellis),
# tensor-core-shuffle word at v = 0, transcribed in v1ValueIndex below.
# trellisValueIndex must match it exactly.
func v1ValueIndex(kInTile, nInTile: int): int =
  32*(nInTile mod 8) + 8*((kInTile mod 8) div 2) + (kInTile mod 2) +
    2*(kInTile div 8) + 4*(nInTile div 8)

proc checkValueOrder =
  ## Stage 0, host-only: index bijection, coords inverse, v1 agreement.
  var seen: array[256, bool]
  for k in 0 ..< 16:
    for n in 0 ..< 16:
      let i = trellisValueIndex(k, n)
      doAssert i >= 0 and i < 256 and not seen[i],
        &"trellisValueIndex({k}, {n}) = {i} collides or is out of range"
      seen[i] = true
      doAssert trellisValueCoords(i) == (k, n),
        &"trellisValueCoords does not invert trellisValueIndex at ({k}, {n})"
      doAssert trellisValueIndex(k, n) == v1ValueIndex(k, n),
        &"trellisValueIndex disagrees with the v1 formula at ({k}, {n})"
  echo "stage 0 value order: index bijective, inverse of trellisValueCoords, matches v1"

proc checkWindowRoundtrip =
  ## Stage 0, host-only: pack.cu-format streams roundtrip through the dq
  ## funnel window read:
  ##
  ## - the window's low `bits` bits recover the packed code
  ## - every index, every bits level, every tile of a multi-tile stream
  var rng = initRand(0xC17D44'i64)
  staticFor bits, 1, 9:
    var stored = newSeq[uint16](3 * 16 * bits)
    var allCodes: array[3, array[256, uint16]]
    for tile in 0 ..< 3:
      for i in 0 ..< 256:
        allCodes[tile][i] = uint16(rng.rand(0 ..< (1 shl bits)))
      let packed = packCodes(allCodes[tile], bits)
      for j in 0 ..< 16 * bits:
        stored[tile * 16 * bits + j] = packed[j]
    for tile in 0 ..< 3:
      for i in 0 ..< 256:
        let w = refWindow(stored, tile * 16 * bits, i, bits)
        doAssert (w and uint32((1'u32 shl bits) - 1'u32)) == uint32(allCodes[tile][i]),
          "window roundtrip bits=" & $bits & " tile=" & $tile & " index=" & $i &
          ": code not recovered"
  echo "stage 0 windows: pack.cu-format roundtrip recovers every code, bits 1..8, tiles 0..2"

proc checkDequant(engine: HwEngine, rng: var Rand, numTiles: int, bits: int, cb: Exl3Codebook) =
  ## Dequant replay, bit-exact against the reference.
  let
    trellis = makeTrellis(rng, numTiles, bits)
    got = engine.launchDequant(trellis, numTiles, bits, cb)
    want = dequantRef(trellis, numTiles, bits, cb)
  var bad = 0
  for i in 0 ..< want.len:
    if got[i] != want[i]:
      inc bad
  doAssert bad == 0,
    &"dequant bits={bits} cb={cb}: {bad}/{want.len} elements differ from the reference"

proc checkGemm(engine: HwEngine, rng: var Rand, m, k, n: int,
    bits: int, cb: Exl3Codebook, OutT: typedesc, withSuh: static bool,
    rtol, abstol: float64, label: string, gridTiles: int = 0, gridM: int = 0) =
  ## GEMM replay, banded against the float64 reference.
  let
    aBits = makeF16Tensor(rng, -2.0'f32, 2.0'f32, m, k)
    trellis = makeTrellis(rng, (k div 16) * (n div 16), bits)
    suh = when withSuh: makeF16Tensor(rng, 0.25'f32, 1.75'f32, 1, k) else: newSeq[uint16](0)
    got = engine.launchGemm(aBits, trellis, m, k, n, bits, cb, OutT, suh, withSuh, gridTiles, gridM)
  var want: seq[float64]
  when withSuh:
    # Input scaling pass in the reference, then the GEMM over the transformed activations
    let had = newSeq[uint16](m * k)
    hadamardRef(
      cast[ptr UncheckedArray[float16]](addr had[0]),
      cast[ptr UncheckedArray[float16]](unsafeAddr aBits[0]),
      cast[ptr UncheckedArray[float16]](addr suh[0]),
      int32(m), int32(k), exl3Norm128)
    want = gemmRef(had, trellis, m, k, n, bits, cb)
  else:
    want = gemmRef(aBits, trellis, m, k, n, bits, cb)
  bandedCheck(got, want, rtol, abstol, label)

proc main =
  var engine = newEngine(runMetal)
  var rng = initRand(0xC17D44'i64)

  # Stage 0: decode contract properties, host-only
  checkValueOrder()
  checkWindowRoundtrip()

  # Stage 1: dequant, bit-exact, every bits level and codebook
  staticFor bits, 1, 9:
    checkDequant(engine, rng, 5, bits, cb3inst)
  checkDequant(engine, rng, 5, 2, cbMcg)
  checkDequant(engine, rng, 5, 4, cbMcg)
  checkDequant(engine, rng, 5, 2, cbMul1)
  checkDequant(engine, rng, 5, 4, cbMul1)
  echo "stage 1 dequant: bit-exact vs reference on all bits and codebooks"

  # Stage 2: GEMM, smallest shape first
  checkGemm(engine, rng, 16, 32, 128, 2, cb3inst, float32, false, 1e-3, 1e-3, "bits2 one chunk fp32")
  checkGemm(engine, rng, 16, 32, 128, 2, cb3inst, float16, false, 2e-3, 5e-2, "bits2 one chunk fp16")
  checkGemm(engine, rng, 16, 32, 128, 4, cb3inst, float16, false, 2e-3, 5e-2, "bits4 one chunk fp16")
  checkGemm(engine, rng, 37, 64, 128, 2, cb3inst, float16, false, 2e-3, 5e-2, "bits2 multi-chunk fp16")
  checkGemm(engine, rng, 16, 64, 256, 4, cb3inst, float16, false, 2e-3, 5e-2, "bits4 n256 fp16")
  checkGemm(engine, rng, 16, 32, 128, 4, cbMcg, float32, false, 1e-3, 1e-3, "bits4 mcg fp32")
  checkGemm(engine, rng, 16, 32, 128, 4, cbMul1, float16, false, 2e-3, 5e-2, "bits4 mul1 fp16")
  checkGemm(engine, rng, 16, 128, 128, 2, cb3inst, float16, true, 2e-3, 5e-2, "bits2 suh fp16")
  checkGemm(engine, rng, 16, 512, 128, 2, cb3inst, float16, false, 2e-3, 5e-2, "bits2 k512 fp16")
  checkGemm(engine, rng, 16, 64, 256, 4, cb3inst, float16, false, 2e-3, 5e-2, "bits4 n256 undersized grid", 1)
  checkGemm(engine, rng, 37, 64, 128, 2, cb3inst, float16, false, 2e-3, 5e-2, "bits2 multi-chunk undersized grid y", 0, 1)
  checkGemm(engine, rng, 37, 64, 128, 2, cb3inst, float16, false, 2e-3, 5e-2, "bits2 multi-chunk undersized both axes", 1, 1)
  echo "T_EXL3_GEMM_METAL VERDICT: dequant bit-exact, GEMM inside band vs f64 reference on all cases"

main()
