## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/wip \
##   --nimcache:nimcache/wip \
##   workspace/positron/tests/manual_gguf_dequant_fp16.nim

import std/strformat, workspace/crucible, workspace/ceramic, workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils, ../src/kernels/ceramic/gguf_ops, ./gguf_test_utils

const ggufDequantMsl = metal:
  proc dequantTile[A: static MmaAtom](bReg: var RtRight[float16, 16, 32, A],
      Out: ptr UncheckedArray[float16], w: ptr UncheckedArray[uint8], K, rowBytes, scheme: int32) {.device.} =
    let kk = int32(threadgroup_position_in_grid.x)
    let tile = int32(threadgroup_position_in_grid.y)
    if scheme == 0:
      dequantGGUF_Q8_0(bReg, w, kk, rowBytes, tile div 4, tile mod 4)
    else:
      if scheme == 1:
        dequantGGUF_Q4_K(bReg, w, kk, rowBytes, tile div 4, tile mod 4)
      else:
        dequantGGUF_IQ4_XS(bReg, w, kk, rowBytes, tile div 4, tile mod 4)
    let cell = crd2idx(A.getLayoutA(), (int(thread_index_in_threadgroup), 0)).toIntVal()
    for n in 0'i32 ..< 2:
      for m in 0'i32 ..< 4:
        for v in 0'i32 ..< 2:
          Out[(tile * 32 + (cell div 8) + 8 * m + v) * K + kk * 16 + (cell mod 8) + 8 * n] =
            bReg.frags[m][n].frag[v]
  proc dequantQ8_0(Out: ptr UncheckedArray[float16], w: ptr UncheckedArray[uint8], K, rowBytes: int32) {.global.} =
    var bReg: rt_r(float16, 16, 32)
    dequantTile(bReg, Out, w, K, rowBytes, 0)
  proc dequantQ4_K(Out: ptr UncheckedArray[float16], w: ptr UncheckedArray[uint8], K, rowBytes: int32) {.global.} =
    var bReg: rt_r(float16, 16, 32)
    dequantTile(bReg, Out, w, K, rowBytes, 1)
  proc dequantIQ4_XS(Out: ptr UncheckedArray[float16], w: ptr UncheckedArray[uint8], K, rowBytes: int32) {.global.} =
    var bReg: rt_r(float16, 16, 32)
    dequantTile(bReg, Out, w, K, rowBytes, 2)

proc dequantKernel(engine: var auto, launcher: string, packed: seq[uint8],
    K, N, rowBytes: int): seq[uint16] =
  var outO = newSeq[uint16](N * K)
  engine.run << (grid: (K div 16, N div 32, 1), blk: (32, 1)) >> (
    launcher, outO, (packed, int32(K), int32(rowBytes)))
  result = outO

func worstAbsDiff(a, b: seq[uint16]): float32 =
  result = 0.0'f32
  for i in 0 ..< a.len:
    let d = abs(fp16ToFp32(a[i]) - fp16ToFp32(b[i]))
    if d > result: result = d

proc checkGGUFDequant(): bool =
  var engine = bkMetal.init()
  engine.ingest(ggufDequantMsl)
  let cases = [
    ("dequantQ8_0", 128, 128, genQ8_0, decodeWeightsQ8_0),
    ("dequantQ8_0", 256, 256, genQ8_0, decodeWeightsQ8_0),
    ("dequantQ4_K", 256, 256, genQ4_K, decodeWeightsQ4_K),
    ("dequantQ4_K", 512, 128, genQ4_K, decodeWeightsQ4_K),
    ("dequantIQ4_XS", 256, 256, genIQ4_XS, decodeWeightsIQ4_XS),
    ("dequantIQ4_XS", 512, 128, genIQ4_XS, decodeWeightsIQ4_XS),
  ]
  var worstAll = 0.0'f32
  for d in 0 ..< cases.len:
    let (launcher, K, N, gen, decode) = cases[d]
    let packed = gen(K, N, 100 + d)
    let actual = dequantKernel(engine, launcher, packed, K, N, packed.len div N)
    let expected = decode(packed, K, N)
    let w = worstAbsDiff(actual, expected)
    if w > worstAll: worstAll = w
    echo &"  {launcher} K={K} N={N}: worst |Δ| = {w}, {actual.len} elements"
    doAssert actual == expected, &"{launcher} K={K} N={N}: bit mismatch"
  echo &"  worst |Δ| across {cases.len} cases = {worstAll} (bit-exact)"
  result = true

when isMainModule:
  runCppTest("GGUF dequant Q8_0/Q4_K/IQ4_XS vs the shared reconstruction", checkGGUFDequant)
