## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/wip \
##   --nimcache:nimcache/wip \
##   workspace/positron/tests/manual_causal_conv1d_bf16.nim

import std/random
import std/strformat
import workspace/crucible
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
from workspace/libtorch/src/raw_libtorch import manual_seed
import ../src/kernels/ceramic/causal_conv1d
import ./bf16_test_utils

const conv1dMsl = metal:
  proc conv1dFwdOut(Out: ptr UncheckedArray[bfloat16], X: ptr UncheckedArray[bfloat16],
      StateOut: ptr UncheckedArray[bfloat16], W: ptr UncheckedArray[bfloat16],
      H, T: int32) {.global.} =
    causal_conv1d_fwd(Out, X, StateOut, W, H, T, 16, 3)
  proc conv1dFwdState(StateOut: ptr UncheckedArray[bfloat16], X: ptr UncheckedArray[bfloat16],
      Out: ptr UncheckedArray[bfloat16], W: ptr UncheckedArray[bfloat16],
      H, T: int32) {.global.} =
    causal_conv1d_fwd(Out, X, StateOut, W, H, T, 16, 3)

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

proc conv1dFwdKernel(xBits, wBits: seq[uint16], H, T: int): tuple[outO, stateO: F.Tensor] =
  ## Calls causal_conv1d_fwd twice, once to read back the conv output,
  ## once to read back the state-out. The engine reads back only binding 0,
  ## hence one launch per output.
  var engine = bkMetal.init()
  engine.ingest(conv1dMsl)
  var outO = newSeq[uint16](H * T)
  var stateO = newSeq[uint16](H * 2)
  engine.run << (grid: (T div 16, (H + 7) div 8, 1), blk: (32, 1)) >> (
    "conv1dFwdOut", outO, (xBits, stateO, wBits, int32(H), int32(T)))
  engine.run << (grid: (T div 16, (H + 7) div 8, 1), blk: (32, 1)) >> (
    "conv1dFwdState", stateO, (xBits, outO, wBits, int32(H), int32(T)))
  result = (toTensor(bitsToF32(outO)).reshape(H, T),
            toTensor(bitsToF32(stateO)).reshape(H, 2))

proc conv1dFwdReference(xBits, wBits: seq[uint16], H, T, K: int): tuple[outO, stateO: F.Tensor] =
  ## Torch reference over the same bf16-rounded inputs.
  ##
  ## - output: padded depthwise conv1d narrowed to the first T columns
  ## - state: the last K-1 input columns
  let x = toTensor(bitsToF32(xBits)).reshape(H, T).to(kBfloat16)
  let w = toTensor(bitsToF32(wBits)).reshape(H, K).to(kBfloat16)
  let conv = F.conv1d(x.reshape(1, H, T), w.reshape(H, 1, K), padding = [K - 1], groups = H)
  let outT = conv.narrow(2, 0, T).reshape(H, T).to(kFloat32)
  let stateT = x.narrow(1, T - (K - 1), K - 1).reshape(H, K - 1).to(kFloat32)
  result = (outT, stateT)

proc checkFwd(): bool =
  Torch.manual_seed(0x5EED'u64)
  var rng = initRand(0x5EED)
  const K = 3
  for (H, T) in [(1024, 64), (24, 32), (64, 128), (20, 32), (7, 32)]:
    let xBits = randomBf16Buffer(rng, H * T, 0.5'f32)
    let wBits = randomBf16Buffer(rng, H * K, 1.0'f32)
    let (actual, actualState) = conv1dFwdKernel(xBits, wBits, H, T)
    let (expected, expectedState) = conv1dFwdReference(xBits, wBits, H, T, K)
    echo &"  fwd H={H} T={T}: worst |Δ| out = {worstAbsDiff(actual, expected)}, state = {worstAbsDiff(actualState, expectedState)} (tolerance 5e-3)"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
    assertAllClose(actualState, expectedState, rtol = 0.0'f64, abstol = 5e-3'f64)
  result = true

when isMainModule:
  runCppTest("causal_conv1d_fwd vs the torch reference", checkFwd)
