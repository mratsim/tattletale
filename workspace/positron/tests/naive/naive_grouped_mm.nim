# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## grouped_mm reference for the positron naive test tier, the torch
## `at::_grouped_mm` op spelled fresh in plain Nim
## design input, the taxonomy consumer transformers/src/layers/ffn.nim:684-784
##
##   out[r, i] = El(sum_h a[r, h] · mat2ᵀ[group(r), h, i])   fp32 accumulate
##   El = one round-to-nearest-even round in the family dtype
##
## | term     | contract                                                                                                       |
## | -------- | -------------------------------------------------------------------------------------------------------------- |
## | call     | (a, mat2ᵀ, offs), rows of `a` pair-major grouped by expert id, group g covering rows offs[g-1] ..< offs[g]     |
## | offs     | inclusive per-expert end offsets, non-decreasing, the last entry closes the row range, a repeat an empty group |
## | w        | the pre-transpose cube [E, I, H], the transpose in the call is convention, mat2ᵀ (e, h, i) reads w[e, i, h]    |
## | rounding | one fp32 sequential accumulation per output element, one El round at the store                                 |
##
## Consumers, the MoE stage of the mega decode kernel judges its grouped GEMM
## against `naiveGroupedMm`, the self-test is t_naive_grouped_mm.nim

import naive_tensors

type GmmFamily* = enum
  ## 16-bit storage families the grouped GEMM rounds into at the store.
  gmmBf16, gmmF16

proc gmmWiden*(fam: GmmFamily, h: uint16): float32 =
  ## Returns the exact fp32 widening of a family-dtype bit pattern.
  if fam == gmmBf16: bf16ToF32(h) else: fp16ToFp32(h)

proc gmmRoundEl*(fam: GmmFamily, x: float32): uint16 =
  ## Returns the family-dtype round-to-nearest-even bit pattern of an fp32 value,
  ## the one El round at the grouped GEMM's store.
  if fam == gmmBf16: f32ToBf16(x) else: fp32ToFp16(x)

proc gmmName*(fam: GmmFamily): string =
  ## Returns the family dtype's display name.
  if fam == gmmBf16: "bf16" else: "fp16"

proc groupOfRow*(offs: seq[int32], row: int): int =
  ## Returns the expert group owning `row` under the inclusive end-offset contract
  ## -1 for a row past the last offset, a caller bug
  doAssert offs.len > 0 and row >= 0 and row < offs[^1].int,
    "row outside the offsets coverage"
  for g in 0 ..< offs.len:
    if row < offs[g].int:
      return g
  doAssert false, "unreachable"

proc naiveGroupedMm*(fam: GmmFamily; a: NaiveMat[uint16]; w: NaiveCube[uint16];
    offs: seq[int32]): NaiveMat[uint16] =
  ## Grouped GEMM over expert groups, one fp32 accumulation and one El
  ## round per output element.
  ##
  ## Expected input:
  ## - `a`, (P, H) family-dtype rows, pair-major grouped by expert id,
  ##   group g covering rows offs[g-1] ..< offs[g]
  ## - `w`, (E, I, H) family-dtype pre-transpose cube, element (e, h, i)
  ##   of mat2ᵀ reading w[e, i, h]
  ## - `offs`, E inclusive end offsets, non-decreasing, offs[E-1] = P,
  ##   a repeated entry an empty group
  ##
  ## Output:
  ## - (P, I) family-dtype rows, out[r, i] = El(sum_h a[r, h]·w[e, i, h])
  ##   over the row's group e, fp32 sequential accumulation
  ##
  ## Example (H = 2, exact small values, bf16):
  ##   a[0] = [1.0, 2.0], w[e, 0, :] = [0.5, 0.25] → out[0, 0] = 1.0
  let rows = a.rows
  doAssert offs.len == w.planes, "one offset per expert"
  doAssert offs[^1].int == rows, "the last offset must close the row range"
  doAssert a.cols == w.cols, "the contraction dim must match the cube's H"
  result = NaiveMat[uint16](rows: rows, cols: w.rows)
  result.data = newSeq[uint16](rows * w.rows)
  for e in 0 ..< w.planes:
    let lo = (if e == 0: 0 else: offs[e - 1].int)
    let hi = offs[e].int
    for r in lo ..< hi:
      for i in 0 ..< w.rows:
        var acc = 0.0'f32
        for h in 0 ..< a.cols:
          acc += gmmWiden(fam, a.data[r * a.cols + h]) *
            gmmWiden(fam, w.data[(e * w.rows + i) * w.cols + h])
        result.data[r * w.rows + i] = gmmRoundEl(fam, acc)

proc naiveGroupedMmSums*(fam: GmmFamily; a: NaiveMat[uint16]; w: NaiveCube[uint16];
    offs: seq[int32]): NaiveMat[float32] =
  ## Returns the unrounded fp32 accumulations of `naiveGroupedMm`, same contract,
  ## for checks that need the pre-store value.
  doAssert a.cols == w.cols, "the contraction dim must match the cube's H"
  let rows = a.rows
  doAssert offs.len == w.planes, "one offset per expert"
  doAssert offs[^1].int == rows, "the last offset must close the row range"
  result = NaiveMat[float32](rows: rows, cols: w.rows)
  result.data = newSeq[float32](rows * w.rows)
  for e in 0 ..< w.planes:
    let lo = (if e == 0: 0 else: offs[e - 1].int)
    let hi = offs[e].int
    for r in lo ..< hi:
      for i in 0 ..< w.rows:
        var acc = 0.0'f32
        for h in 0 ..< a.cols:
          acc += gmmWiden(fam, a.data[r * a.cols + h]) *
            gmmWiden(fam, w.data[(e * w.rows + i) * w.cols + h])
        result.data[r * w.rows + i] = acc
