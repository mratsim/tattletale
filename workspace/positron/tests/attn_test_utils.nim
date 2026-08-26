## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared attention-test support: the fp16-rounded random input
## generator, the fp16 seq converters, the worst-difference metric and
## the NEOX rotary tables that the manual attention tests share.

import std/math
import workspace/libtorch
import workspace/libtorch as F
import ../../ceramic/tests/tile_test_utils

proc scaledRand*(rows, cols: int, scaleF: float32): seq[float32] =
  ## Returns rand(rows, cols) shifted host-side to [-scaleF, scaleF):
  ## the kernel buffer and the reference tensors derive from the same
  ## seq.
  let t = rand(rows, cols)
  let p = t.contiguous().data_ptr(float32)
  result = newSeq[float32](rows * cols)
  for i in 0 ..< rows * cols:
    result[i] = (p[i] - 0.5'f32) * (2.0'f32 * scaleF)

proc worstAbsDiff*(a, b: F.Tensor): float32 =
  ## Returns the largest |a[i] - b[i]| over all elements.
  let ap = a.contiguous().data_ptr(float32)
  let bp = b.contiguous().data_ptr(float32)
  result = 0.0'f32
  for i in 0 ..< a.numel():
    let d = abs(ap[i] - bp[i])
    if d > result: result = d

proc fp16sToF32*(hs: seq[uint16]): seq[float32] =
  ## Widens an fp16 bit-pattern buffer to fp32 values.
  result = newSeq[float32](hs.len)
  for i in 0 ..< hs.len:
    result[i] = fp16ToFp32(hs[i])

proc fp32sToFp16*(fs: seq[float32]): seq[uint16] =
  ## Rounds an fp32 value buffer down to fp16 bit patterns (RNE).
  result = newSeq[uint16](fs.len)
  for i in 0 ..< fs.len:
    result[i] = fp32ToFp16(fs[i])

proc buildRopeCosSin*(rows, dim: int, theta: float32): tuple[cosT, sinT: seq[float32]] =
  ## Returns the NEOX fp32 cos/sin tables: row r carries cos/sin of
  ## inv_freq[t]·r with inv_freq[t] = theta^(−2t/dim), t in
  ## [0, dim/2).
  let half = dim shr 1
  result.cosT = newSeq[float32](rows * half)
  result.sinT = newSeq[float32](rows * half)
  for r in 0 ..< rows:
    for t in 0 ..< half:
      let ang = pow(float64(theta), -float64(2 * t) / float64(dim)).float32 * float32(r)
      result.cosT[r * half + t] = cos(ang)
      result.sinT[r * half + t] = sin(ang)
