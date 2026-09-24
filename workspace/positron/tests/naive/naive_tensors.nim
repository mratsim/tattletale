# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Small-shape random tensors for the positron naive test tier.
## Row-major matrices and 3-D cubes over owned seq storage, filled
## from a seeded `NaiveRng`.
##
## Also carried here:
## - bf16 bit plumbing (exact widen, round-to-nearest-even narrow)
## - fp16/bf16 narrowing and widening pairs for kernel-vs-reference comparisons
##
## 16-bit dtypes are stored as their IEEE-754 bit patterns (uint16),
## the shape kernels read. float32 storage appears where arithmetic
## or a metric needs the widened values.

import naive_rng
from ../../../ceramic/tests/tile_test_utils import fp32ToFp16, fp16ToFp32
export fp32ToFp16, fp16ToFp32

# bf16 bit plumbing, round-to-nearest-even both directions

func bf16ToF32*(h: uint16): float32 =
  ## Returns the exact bfloat16 → binary32 widening of the pattern.
  ## A bf16 pattern is the high half of an fp32 pattern, so widening is
  ## a 16-bit left shift. Subnormals, Inf and NaN widen exactly.
  cast[float32](uint32(h) shl 16)

func f32ToBf16*(x: float32): uint16 =
  ## Returns the IEEE-754 binary32 → bfloat16 bit pattern of the value,
  ## round-to-nearest-even. Zero and Inf pass through.
  ##
  ## A NaN input keeps a NaN pattern, the canonical quiet NaN.
  ## Payload is not preserved, the sign is. Overflow rounds to Inf.
  let u = cast[uint32](x)
  if (u and 0x7F800000'u32) == 0x7F800000'u32 and (u and 0x007FFFFF'u32) != 0:
    # The round-to-nearest increment can carry across the Inf/NaN
    # boundary (a small odd payload rounds a NaN to Inf), so a NaN
    # short-circuits to the canonical quiet NaN, sign kept.
    return if (u shr 31) == 1: 0xFFC0'u16 else: 0x7FC0'u16
  uint16((u + 0x7FFF'u32 + ((u shr 16) and 1)) shr 16)

# Owned-storage tensor types

type
  NaiveMat*[T] = object
    ## Row-major M×N matrix over owned storage.
    ## - element (r, c) lives at data[r * cols + c]
    rows*, cols*: int
    data*: seq[T]

  NaiveCube*[T] = object
    ## Row-major P×R×C tensor over owned storage.
    ## - element (p, r, c) lives at data[(p * rows + r) * cols + c]
    planes*, rows*, cols*: int
    data*: seq[T]

func at*[T](m: var NaiveMat[T]; r, c: int): var T =
  ## Returns element (r, c) as a mutable reference. Both indices must
  ## sit inside the shape, out-of-shape access is a caller bug.
  m.data[r * m.cols + c]

func at*[T](m: var NaiveCube[T]; p, r, c: int): var T =
  ## Returns element (p, r, c) as a mutable reference, with all three
  ## indices inside the shape.
  m.data[(p * m.rows + r) * m.cols + c]

func asUnchecked*[T](s: var seq[T]): ptr UncheckedArray[T] =
  ## Returns the seq storage as an unchecked pointer view, the shape
  ## kernels take. The seq must outlive every use of the pointer,
  ## the test owns it for the whole run.
  doAssert s.len > 0, "cannot view an empty seq"
  cast[ptr UncheckedArray[T]](s[0].unsafeAddr)

# Random init helpers

proc randomMat*(rng: var NaiveRng; rows, cols: int; lo, hi: float32): NaiveMat[float32] =
  ## Returns an M×N matrix of uniform float32 values in [lo, hi).
  result.rows = rows
  result.cols = cols
  result.data = newSeq[float32](rows * cols)
  for i in 0 ..< result.data.len:
    result.data[i] = rng.nextF32(lo, hi)

proc randomCube*(rng: var NaiveRng; planes, rows, cols: int; lo, hi: float32): NaiveCube[float32] =
  ## Returns a P×R×C tensor of uniform float32 values in [lo, hi).
  result.planes = planes
  result.rows = rows
  result.cols = cols
  result.data = newSeq[float32](planes * rows * cols)
  for i in 0 ..< result.data.len:
    result.data[i] = rng.nextF32(lo, hi)

# Dtype conversions, f32 ↔ fp16/bf16 patterns, round-to-nearest-even

proc mapElemsMat[A, B](m: NaiveMat[A]; f: proc (x: A): B {.nimcall.}): NaiveMat[B] =
  result.rows = m.rows
  result.cols = m.cols
  result.data = newSeq[B](m.data.len)
  for i in 0 ..< m.data.len:
    result.data[i] = f(m.data[i])

proc mapElemsCube[A, B](c: NaiveCube[A]; f: proc (x: A): B {.nimcall.}): NaiveCube[B] =
  result.planes = c.planes
  result.rows = c.rows
  result.cols = c.cols
  result.data = newSeq[B](c.data.len)
  for i in 0 ..< c.data.len:
    result.data[i] = f(c.data[i])

proc narrowF16*(m: NaiveMat[float32]): NaiveMat[uint16] =
  ## fp32 values → fp16 bit patterns, round-to-nearest-even.
  mapElemsMat(m, fp32ToFp16)

proc narrowBf16*(m: NaiveMat[float32]): NaiveMat[uint16] =
  ## fp32 values → bfloat16 bit patterns, round-to-nearest-even.
  mapElemsMat(m, f32ToBf16)

proc widenF16*(m: NaiveMat[uint16]): NaiveMat[float32] =
  ## fp16 bit patterns → exact fp32 values.
  mapElemsMat(m, fp16ToFp32)

proc widenBf16*(m: NaiveMat[uint16]): NaiveMat[float32] =
  ## bfloat16 bit patterns → exact fp32 values.
  mapElemsMat(m, bf16ToF32)

proc narrowF16*(c: NaiveCube[float32]): NaiveCube[uint16] =
  ## fp32 values → fp16 bit patterns, round-to-nearest-even.
  mapElemsCube(c, fp32ToFp16)

proc narrowBf16*(c: NaiveCube[float32]): NaiveCube[uint16] =
  ## fp32 values → bfloat16 bit patterns, round-to-nearest-even.
  mapElemsCube(c, f32ToBf16)

proc widenF16*(c: NaiveCube[uint16]): NaiveCube[float32] =
  ## fp16 bit patterns → exact fp32 values.
  mapElemsCube(c, fp16ToFp32)

proc widenBf16*(c: NaiveCube[uint16]): NaiveCube[float32] =
  ## bfloat16 bit patterns → exact fp32 values.
  mapElemsCube(c, bf16ToF32)


# fp64 tensor preparation for the delta-rule tier, fp64 spellings compare
# against each other over the same widened inputs and per-run state copies

proc widenF64*[T: float32|float64](m: NaiveMat[T]): NaiveMat[float64] =
  ## Returns the exact fp64 widening of the matrix, element for element.
  ## The f32 to f64 widening is lossless, so an fp32 run and an fp64 run
  ## fed from this widening see identical values.
  result.rows = m.rows
  result.cols = m.cols
  result.data = newSeq[float64](m.data.len)
  for i in 0 ..< m.data.len:
    result.data[i] = float64(m.data[i])

proc widenF64*[T: float32|float64](t: NaiveCube[T]): NaiveCube[float64] =
  ## Returns the exact fp64 widening of the cube, element for element.
  result.planes = t.planes
  result.rows = t.rows
  result.cols = t.cols
  result.data = newSeq[float64](t.data.len)
  for i in 0 ..< t.data.len:
    result.data[i] = float64(t.data[i])

proc copyTensor*[T](m: NaiveMat[T]): NaiveMat[T] =
  ## Returns an independent copy of the matrix.
  result.rows = m.rows
  result.cols = m.cols
  result.data = newSeq[T](m.data.len)
  for i in 0 ..< m.data.len:
    result.data[i] = m.data[i]

proc copyTensor*[T](t: NaiveCube[T]): NaiveCube[T] =
  ## Returns an independent copy of the cube.
  result.planes = t.planes
  result.rows = t.rows
  result.cols = t.cols
  result.data = newSeq[T](t.data.len)
  for i in 0 ..< t.data.len:
    result.data[i] = t.data[i]

# Run-width casts for suites that execute one comparison at fp32 and fp64

proc castCube*[F: float32|float64](t: NaiveCube[float32]): NaiveCube[F] =
  ## Returns the cube at the run's float width, an independent copy when
  ## F is fp32 and the exact widening when F is fp64, so both dtype runs
  ## of one comparison see identical values.
  when F is float32:
    result = t.copyTensor()
  else:
    result = t.widenF64()

proc castMat*[F: float32|float64](t: NaiveMat[float32]): NaiveMat[F] =
  ## Returns the matrix at the run's float width, same contract as castCube.
  when F is float32:
    result = t.copyTensor()
  else:
    result = t.widenF64()

proc zerosCube*[F: float32|float64](p, r, c: int): NaiveCube[F] =
  ## Returns a zero cube of the given shape at the run's float width.
  NaiveCube[F](planes: p, rows: r, cols: c, data: newSeq[F](p * r * c))

proc fillMat*[T](m: var NaiveMat[T]; value: T) =
  ## Overwrites every matrix element with `value`.
  for i in 0 ..< m.data.len:
    m.data[i] = value

proc fillCube*[T](t: var NaiveCube[T]; value: T) =
  ## Overwrites every cube element with `value`.
  for i in 0 ..< t.data.len:
    t.data[i] = value
