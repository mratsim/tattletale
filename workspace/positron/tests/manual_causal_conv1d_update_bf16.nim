## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/wip \
##   --nimcache:nimcache/wip \
##   workspace/positron/tests/manual_causal_conv1d_update_bf16.nim

import std/random
import std/strformat
import workspace/crucible
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
from workspace/libtorch/src/raw_libtorch import manual_seed
import ../src/kernels/ceramic/causal_conv1d
import ./bf16_test_utils

const convUpdateMsl = metal:
  proc conv1dUpdateOut(Out: ptr UncheckedArray[bfloat16], Win: ptr UncheckedArray[bfloat16],
      StateOut: ptr UncheckedArray[bfloat16], W: ptr UncheckedArray[bfloat16],
      H: int32) {.global.} =
    causal_conv1d_update(Out, Win, StateOut, W, H, 8, 3)
  proc conv1dUpdateState(StateOut: ptr UncheckedArray[bfloat16], Win: ptr UncheckedArray[bfloat16],
      Out: ptr UncheckedArray[bfloat16], W: ptr UncheckedArray[bfloat16],
      H: int32) {.global.} =
    causal_conv1d_update(Out, Win, StateOut, W, H, 8, 3)

proc worstAbsDiff(a, b: F.Tensor): float32 =
  ## Largest |a[i] - b[i]| over all elements.
  let ap = a.contiguous().data_ptr(float32)
  let bp = b.contiguous().data_ptr(float32)
  result = 0.0'f32
  for i in 0 ..< a.numel():
    let d = abs(ap[i] - bp[i])
    if d > result: result = d

proc bitsToF32(bits: seq[uint16]): seq[float32] =
  result = newSeq[float32](bits.len)
  for i in 0 ..< bits.len:
    result[i] = bf16BitsToF32(bits[i])

proc conv1dUpdateKernel(winBits, wBits: seq[uint16], H: int): tuple[outO, stateO: F.Tensor] =
  ## Calls causal_conv1d_update twice, once to read back the decode output
  ## and once to read back the state-out. The engine reads back only
  ## binding 0, hence one launch per output.
  var engine = bkMetal.init()
  engine.ingest(convUpdateMsl)
  var outO = newSeq[uint16](H)
  var stateO = newSeq[uint16](H * 2)
  engine.run << (grid: (1, (H + 7) div 8, 1), blk: (32, 1)) >> (
    "conv1dUpdateOut", outO, (winBits, stateO, wBits, int32(H)))
  engine.run << (grid: (1, (H + 7) div 8, 1), blk: (32, 1)) >> (
    "conv1dUpdateState", stateO, (winBits, outO, wBits, int32(H)))
  result = (toTensor(bitsToF32(outO)).reshape(H, 1),
            toTensor(bitsToF32(stateO)).reshape(H, 2))

proc conv1dUpdateReference(winBits, wBits: seq[uint16], H, K: int): tuple[outO, stateO: F.Tensor] =
  ## Torch reference over the same bf16-rounded inputs.
  ##
  ## - output: depthwise conv1d of the (1, H, K) window, stride 1, no padding
  ## - state: the window's columns 1..K-1
  let win = toTensor(bitsToF32(winBits)).reshape(H, K).to(kBfloat16)
  let w = toTensor(bitsToF32(wBits)).reshape(H, K).to(kBfloat16)
  let conv = F.conv1d(win.reshape(1, H, K), w.reshape(H, 1, K), groups = H)
  let outT = conv.reshape(H, 1).to(kFloat32)
  let stateT = win.narrow(1, 1, K - 1).reshape(H, K - 1).to(kFloat32)
  result = (outT, stateT)

proc checkUpdate(): bool =
  Torch.manual_seed(0x5EED'u64)
  var rng = initRand(0x5EED)
  const K = 3
  for H in [1024, 24, 64, 20, 7]:
    let winBits = randomBf16Buffer(rng, H * K, 0.5'f32)
    let wBits = randomBf16Buffer(rng, H * K, 1.0'f32)
    let (actual, actualState) = conv1dUpdateKernel(winBits, wBits, H)
    let (expected, expectedState) = conv1dUpdateReference(winBits, wBits, H, K)
    echo &"  update H={H}: worst |Δ| out = {worstAbsDiff(actual, expected)}, state = {worstAbsDiff(actualState, expectedState)} (tolerance 5e-3)"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
    assertAllClose(actualState, expectedState, rtol = 0.0'f64, abstol = 5e-3'f64)
  result = true

when isMainModule:
  runCppTest("causal_conv1d_update vs the torch reference", checkUpdate)
