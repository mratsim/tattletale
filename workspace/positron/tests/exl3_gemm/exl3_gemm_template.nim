# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 trellis-dequant GEMM replay: CPU reference, generated fixtures, Metal launchers.
##
## Run from the worktree root:
##   nim c -r --threads:off -o:build/tests/t_exl3_gemm_metal \
##     workspace/positron/tests/exl3_gemm/t_exl3_gemm_metal.nim
##
## Fixtures are generated in the test (random codes, random fp16 activations),
## never hand-written expected values. The reference computes them in float64.
##
## Numerics contract of the comparisons:
##
## - dequant: bit-exact, the decode being 32-bit integer arithmetic plus one
##   round-to-nearest-even to fp16, identical on host and device
## - GEMM fp32: the device accumulates in fp32 through the simdgroup mma,
##   the reference accumulates in float64
## - GEMM fp16: the device rounds its fp32 accumulator to fp16, the reference
##   rounds the float64 sum
##
## Band margins:
##
## - fp32: rtol 1e-3 + abstol 1e-3 over the f64 sums, two orders above the fp32
##   rounding gap at the tested shapes, products bounded by |A| <= 2
##   and |decode| <= 4
## - fp16: rtol 2e-3 + abstol 5e-2, one fp16 ulp of |C| beyond the fp32 band
##   covers a rounding-boundary flip, |C| stays under 2^5 where one fp16 ulp
##   is 2^-10

import std/random
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/positron/src/kernels/exl3_gemm
import workspace/positron/src/kernels/exl3_hadamard_rotate
import ../positron_testutils

# ═══════════════════════════════════════════════════════════════════════
#  CPU reference
# ═══════════════════════════════════════════════════════════════════════

func refWindow*(stored: openArray[uint16], tileU16Base, valueIndex, bits: int): uint32 =
  ## 16-bit decode window of `valueIndex` inside one trellis tile:
  ## exl3_dq.cuh's `dq` funnel over the packed words.
  ##
  ## Contract:
  ##
  ## - the two candidate uint32 words, little-endian pairs of the stored
  ##   uint16 array, merge as (a shl 32) or b and shift right by s0
  ## - words wrap circularly mod 8*bits
  ## - bit-exact transcription of the device kernel's `trellisWindow`
  ##   (exl3_dq.cuh dq extraction) over the pack.cu stream format
  let
    nWords = 8 * bits
    b0 = valueIndex * bits + bits - 16 + 256 * bits
    b1 = b0 + 16
    i0 = b0 div 32
    i1 = (b1 - 1) div 32
    s0 = (i1 + 1) * 32 - b1
    wa = tileU16Base + 2 * (i0 mod nWords)
    wb = tileU16Base + 2 * (i1 mod nWords)
    a = uint32(stored[wa]) or (uint32(stored[wa + 1]) shl 16)
    b = uint32(stored[wb]) or (uint32(stored[wb + 1]) shl 16)
  uint32(((uint64(a) shl 32) or uint64(b)) shr s0 and 0xFFFF'u64)

proc dequantRef*(trellis: openArray[uint16], numTiles, bits: int, cb: Exl3Codebook): seq[uint16] =
  ## Reference dequant of `numTiles` 16x16 tiles into fp16 bits.
  ## Byte-identical with the device kernel.
  ##
  ## Layout: element (k, n) of tile t at offset t*256 + k*16 + n.
  ##
  ## Args:
  ##
  ## - trellis: the packed bitstream, 16*bits uint16 per tile
  ## - numTiles: tiles to dequantize
  ## - bits: bits per value, 1..8
  ## - cb: the codebook
  result = newSeq[uint16](numTiles * 256)
  for t in 0 ..< numTiles:
    for i in 0 ..< 256:
      let
        w = refWindow(trellis, t * 16 * bits, i, bits)
        (k, n) = trellisValueCoords(i)
      result[t * 256 + k * 16 + n] = cast[uint16](decode3inst(w, cb))

proc gemmRef*(aBits, trellis: openArray[uint16], m, k, n, bits: int, cb: Exl3Codebook): seq[float64] =
  ## Reference `C = A @ dequant(B)` in float64.
  ##
  ## Args:
  ##
  ## - aBits: (m, k) fp16 activations, products taken over the exact fp16 values
  ## - trellis: the packed bitstream, dequant per trellisValueIndex
  ## - accumulation in float64, one float64 sum per output element
  let tilesN = n div 16
  var d = newSeq[float64](k * n)
  for kk in 0 ..< k:
    for nn in 0 ..< n:
      let
        tk = kk div 16
        tn = nn div 16
        w = refWindow(trellis, (tk * tilesN + tn) * 16 * bits,
          trellisValueIndex(kk mod 16, nn mod 16), bits)
      d[kk * n + nn] = to(decode3inst(w, cb), float32).float64
  result = newSeq[float64](m * n)
  for mm in 0 ..< m:
    for nn in 0 ..< n:
      var s = 0.0'f64
      for kk in 0 ..< k:
        s += to(cast[float16](aBits[mm * k + kk]), float32).float64 * d[kk * n + nn]
      result[mm * n + nn] = s

func refFwht128*(x: var array[128, float64]) =
  ## Unnormalized 7-stage fast Walsh-Hadamard transform, in place over x.
  var step = 1
  while step < 128:
    var base = 0
    while base < 128:
      for j in 0 ..< step:
        let
          a = x[base + j]
          b = x[base + j + step]
        x[base + j] = a + b
        x[base + j + step] = a - b
      base += 2 * step
    step *= 2

proc hadamardRef*(dst, src, preScale: ptr UncheckedArray[float16], M, C: int32, norm: float32) =
  ## Reference of the input scaling pass `su`, per 128-block:
  ## `dst = fp16round(FWHT(fp16round(src * preScale))) * norm` in float64.
  var row: array[128, float64]
  for i in 0'i32 ..< M:
    for blk in 0'i32 ..< C div 128:
      for kk in 0'i32 ..< 128:
        let v = src[i * C + blk * 128 + kk]
        row[kk] = to(to(v, float32) * to(preScale[blk * 128 + kk], float32), float16).to(float32).float64
      refFwht128(row)
      for kk in 0'i32 ..< 128:
        dst[i * C + blk * 128 + kk] = (row[kk].float32 * norm).to(float16)

# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

proc packCodes*(codes: openArray[uint16], bits: int): seq[uint16] =
  ## Packs one tile's 256 codes into the true exllamav3 stream format
  ## (pack.cu, the exl3_encode pack kernel's transcription).
  ##
  ## Contract:
  ##
  ## - 16 spans of 16 codes pack their low `bits` bits MSB-first into
  ##   `bits` uint16 words per span
  ## - the tile's uint32 words emit as half-swapped pairs of uint16 spans
  ##   2t and 2t+1
  var sPacked = newSeq[uint16](16 * bits)
  for t in 0 ..< 16:
    var
      i = 16 * t
      j = bits * t
      k = 32
      buf = 0'u32
    for n in 0 ..< 16:
      let v = uint32(codes[i]) and ((1'u32 shl bits) - 1'u32)
      k -= bits
      buf = buf or (v shl k)
      if k <= 16:
        sPacked[j] = uint16(buf shr 16)
        buf = buf shl 16
        k += 16
        j += 1
      inc i
  result = newSeq[uint16](16 * bits)
  for t in 0 ..< 8 * bits:
    let s32 = uint32(sPacked[2 * t]) or (uint32(sPacked[2 * t + 1]) shl 16)
    let swapped = (s32 shl 16) or (s32 shr 16)
    result[2 * t] = uint16(swapped and 0xFFFF)
    result[2 * t + 1] = uint16(swapped shr 16)

proc makeTrellis*(rng: var Rand, numTiles, bits: int): seq[uint16] =
  ## Random trellis bitstream over `numTiles` 16x16 tiles, the true exllamav3
  ## pack format: per tile, random codes in ring order pack through `packCodes`.
  result = newSeq[uint16](numTiles * 16 * bits)
  for t in 0 ..< numTiles:
    var codes = newSeq[uint16](256)
    for i in 0 ..< 256:
      codes[i] = uint16(rng.rand(0 ..< (1 shl bits)))
    let packed = packCodes(codes, bits)
    for j in 0 ..< 16 * bits:
      result[t * 16 * bits + j] = packed[j]

proc makeF16Tensor*(rng: var Rand, lo, hi: float32, m, k: int): seq[uint16] =
  ## Random fp16 activation tensor of shape (m, k), contiguous, uniform over [lo, hi].
  result = newSeq[uint16](m * k)
  for i in 0 ..< m * k:
    result[i] = uint16(rng.rand(lo .. hi).to(float16))

# ═══════════════════════════════════════════════════════════════════════
#  Device entries and launchers
# ═══════════════════════════════════════════════════════════════════════

proc exl3DequantCombo*(dst, trellis: distinct TensorView, bits: int, cb: Exl3Codebook) =
  # Requires export: the launch machinery re-emits the callee name into the importing module, an imported-only symbol does not resolve there.
  exl3_dequant_tile(dst, trellis, bits, cb)

proc exl3GemmCombo*(d, a, trellis: distinct TensorView, bits: int, cb: Exl3Codebook) =
  # Requires export: the launch machinery re-emits the callee name into the importing module, an imported-only symbol does not resolve there.
  exl3_gemm(d, a, trellis, bits, cb)

proc hadamardCombo*(dst, src, preScale: distinct TensorView, M, C: int, norm: float32) =
  # Requires export: the launch machinery re-emits the callee name into the importing module, an imported-only symbol does not resolve there.
  # PostScale stays dereferenced only under its static switch, the preScale pointer legally fills that slot here
  hadamard_rotate_128_fwd(
    cast[ptr UncheckedArray[float16]](dst.data), cast[ptr UncheckedArray[float16]](src.data),
    cast[ptr UncheckedArray[float16]](preScale.data), cast[ptr UncheckedArray[float16]](preScale.data),
    int32(M), int32(C), norm, true, false)

proc launchDequant*[E: HwEngine](engine: E, trellis: seq[uint16], numTiles: int, bits: int, cb: Exl3Codebook): seq[uint16] =
  ## One dequant launch over `numTiles` tiles, dst bits read back.
  ##
  ## Args:
  ##
  ## - trellis: uploads as the same byte stream retyped to little-endian
  ##   uint32 words, the kernel's read width
  let
    trellisDev = engine.allocDevice(uint32, trellis.len div 2)
    dstDev = engine.allocDevice(float16, numTiles * 256)
  engine.copyToDeviceFromHost(trellisDev, cast[ptr uint32](addr trellis[0]), trellis.len div 2)
  let
    tLay = make_layout((trellis.len div 2,))
    dLay = make_layout((numTiles * 256,))
    tView = BridgeTensor[typeof(trellisDev), typeof(tLay.shape), typeof(tLay.stride)](
      data: trellisDev, layout: tLay)
    dView = BridgeTensor[typeof(dstDev), typeof(dLay.shape), typeof(dLay.stride)](
      data: dstDev, layout: dLay)
  engine.launchDeviceFn exl3DequantCombo(dView, tView, bits, cb) do:
    grid = (numTiles, 1, 1)
    blk = (32, 1)
  result = newSeq[uint16](numTiles * 256)
  engine.copyToHostFromDevice(cast[ptr float16](addr result[0]), dView.data, numTiles * 256)

proc launchGemm*[E: HwEngine](engine: E, aBits, trellis: seq[uint16], m, k, n: int, bits: int, cb: Exl3Codebook, OutT: typedesc, suh: seq[uint16] = @[], withSuh: static bool = false, gridTiles: int = 0, gridM: int = 0): seq[float32] =
  ## One EXL3 GEMM replay, C read back as fp32 values.
  ##
  ## Args:
  ##
  ## - aBits: (m, k) fp16 activations
  ## - OutT: the output element type, fp32 or fp16, the device store's dtype
  ## - withSuh: run the input scaling pass first over `suh`, the exllamav3 pre-m-loop
  ##   Hadamard rotate writing a scratch copy of the activations
  ## - gridTiles, gridM: threadgroups along the output tiles and m chunks,
  ##   0 selects n div exl3GemmNT and m ceil_div exl3GemmMT, a smaller count
  ##   grid-strides every tile or chunk from one threadgroup
  ##
  ## Every launch bridges fresh buffers, the views move into the launch.
  ## One launch covers the whole m axis, grid.y indexes the kernel's
  ## 16-position m chunks.
  result = newSeq[float32](m * n)
  doAssert k mod 16 == 0, "launchGemm: sizeK not a multiple of 16"
  doAssert n mod exl3GemmNT == 0, "launchGemm: sizeN not a multiple of exl3GemmNT"
  var aSrc = aBits
  when withSuh:
    let
      suhDev = engine.allocDevice(float16, k)
      aInDev = engine.allocDevice(float16, m * k)
      hadDev = engine.allocDevice(float16, m * k)
    engine.copyToDeviceFromHost(aInDev, cast[ptr float16](addr aBits[0]), m * k)
    engine.copyToDeviceFromHost(suhDev, cast[ptr float16](addr suh[0]), k)
    let
      suhLay = make_layout((k,))
      aInLay = make_layout((m, k), (k, 1))
      suhView = BridgeTensor[typeof(suhDev), typeof(suhLay.shape), typeof(suhLay.stride)](
        data: suhDev, layout: suhLay)
      aInView = BridgeTensor[typeof(aInDev), typeof(aInLay.shape), typeof(aInLay.stride)](
        data: aInDev, layout: aInLay)
      hadView = BridgeTensor[typeof(hadDev), typeof(aInLay.shape), typeof(aInLay.stride)](
        data: hadDev, layout: aInLay)
    engine.launchDeviceFn hadamardCombo(hadView, aInView, suhView, m, k, exl3Norm128) do:
      grid = (m, k div 128, 1)
      blk = (32, 1)
    aSrc = newSeq[uint16](m * k)
    engine.copyToHostFromDevice(cast[ptr float16](addr aSrc[0]), hadView.data, m * k)

  let
    aDev = engine.allocDevice(float16, m * k)
    tDev = engine.allocDevice(uint32, trellis.len div 2)
    dDev = engine.allocDevice(OutT, m * n)
    aLay = make_layout((m, k), (k, 1))
    dLay = make_layout((m, n), (n, 1))
    tLay = make_layout((trellis.len div 2,))
  engine.copyToDeviceFromHost(aDev, cast[ptr float16](addr aSrc[0]), m * k)
  engine.copyToDeviceFromHost(tDev, cast[ptr uint32](addr trellis[0]), trellis.len div 2)
  let
    aView = BridgeTensor[typeof(aDev), typeof(aLay.shape), typeof(aLay.stride)](
      data: aDev, layout: aLay)
    tView = BridgeTensor[typeof(tDev), typeof(tLay.shape), typeof(tLay.stride)](
      data: tDev, layout: tLay)
    dView = BridgeTensor[typeof(dDev), typeof(dLay.shape), typeof(dLay.stride)](
      data: dDev, layout: dLay)
  engine.launchDeviceFn exl3GemmCombo(dView, aView, tView, bits, cb) do:
    grid = ((if gridTiles == 0: n div exl3GemmNT else: gridTiles),
            (if gridM == 0: m.ceil_div(exl3GemmMT) else: gridM), 1)
    blk = (32, 1)
  when OutT is float32:
    engine.copyToHostFromDevice(cast[ptr float32](addr result[0]), dView.data, m * n)
  else:
    let outBits = newSeq[uint16](m * n)
    engine.copyToHostFromDevice(cast[ptr float16](addr outBits[0]), dView.data, m * n)
    for i in 0 ..< m * n:
      result[i] = to(cast[float16](outBits[i]), float32)

# ═══════════════════════════════════════════════════════════════════════
#  Comparisons
# ═══════════════════════════════════════════════════════════════════════

proc bandedCheck*(got: openArray[float32], want: openArray[float64], rtol, abstol: float64, label: string) =
  ## |got - want| <= abstol + rtol*|want| per element, worst case reported.
  doAssert got.len == want.len, label & ": length mismatch"
  var
    bad = 0
    worst = 0.0'f64
    worstI = 0
  for i in 0 ..< want.len:
    let
      g = got[i].float64
      w = want[i]
      d = abs(g - w)
    if d > abstol + rtol * abs(w):
      inc bad
      if d > worst:
        worst = d
        worstI = i
  doAssert bad == 0, label & ": " & $bad & "/" & $want.len &
    " elements outside rtol " & $rtol & " + abstol " & $abstol &
    ", worst |d| = " & $worst & " at " & $worstI &
    " (got " & $got[worstI] & " want " & $want[worstI] & ")"

export int_tuples, layout_algebra, tensors, tile_algebra
export exl3GemmMT, exl3GemmNT, exl3Norm128, trellisValueIndex, trellisValueCoords
export Exl3Codebook
export dequantRef, gemmRef, makeTrellis, makeF16Tensor, packCodes, launchDequant, launchGemm,
       bandedCheck, refWindow, hadamardRef
