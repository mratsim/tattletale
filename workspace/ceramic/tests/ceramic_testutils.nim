# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Common utilities for tests over ceramic tensors.
##
## Provides:
## - `assertAllClose` for TensorView, element-wise tolerance check
##   with mismatch diagnostics

import std/[macros, math]
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/fp_conv/fp_conv

# ═══════════════════════════════════════════════════════════════
#  Tensor assertions
# ═══════════════════════════════════════════════════════════════

func toF64(x: float32 | float64 | int | int32 | int64): float64 {.inline.} =
  float64(x)

func toF64(x: F16): float64 {.inline.} =
  float64(float32.from(x))

func flatSize[L](l: L): int =
  ## Cosize as a runtime int, static `Int[N]` layouts included
  when typeof(cosize(l)) is Int:
    toInt(cosize(l))
  else:
    cosize(l)

type AllCloseReport* = object
  bad*: int
  worst*, firstGot*, firstWant*: float64
  firstCrd*: string

proc allCloseImpl[T, ShA, StA, ShB, StB](
    actual: TensorView[T, ShA, StA], expected: TensorView[T, ShB, StB],
    rtol, abstol: float64): AllCloseReport =
  ## Element-wise closeness, |a - e| <= abstol + rtol*|e| per element,
  ## compared in float64, same shape required, strides may differ.
  ## One compiled copy per pair of view types.
  let n = flatSize(actual.getLayout())
  doAssert n == flatSize(expected.getLayout()),
    "allClose: size mismatch, got " & $n &
    " want " & $flatSize(expected.getLayout())
  for i in 0 ..< n:
    let crd = idx2crd(actual.getLayout(), i)
    let a = toF64(actual.data[toInt crd2idx(actual.getLayout(), crd)])
    let e = toF64(expected.data[toInt crd2idx(expected.getLayout(), crd)])
    let d = abs(a - e)
    if d > abstol + rtol * abs(e):
      if result.bad == 0:
        result.firstCrd = $crd
        result.firstGot = a
        result.firstWant = e
      result.worst = max(result.worst, d)
      inc result.bad

template assertAllClose*(actual, expected: untyped, rtol: float64 = 1e-4, abstol: float64 = 1e-4, msg = "") =
  ## Assert two views are element-wise close, compared in float64:
  ## |a - e| <= abstol + rtol*|e| for every element.
  ##
  ## Args:
  ## - actual: the view produced by the code under test
  ## - expected: the reference view, same shape and strides
  ## - rtol, abstol: relative and absolute tolerance
  ## - msg: optional context echoed on failure
  block:
    let actualView = actual
    let expectedView = expected
    let report = allCloseImpl(actualView, expectedView, rtol, abstol)
    if report.bad > 0:
      echo "Assertion failed: allClose"
      echo "  '", astToStr(actual), "'"
      if msg.len > 0:
        echo "  ", msg
      echo "  ", report.bad, " mismatched elements of ",
           cosize(actualView.getLayout()), ", worst |d| = ", report.worst
      echo "  first mismatch at ", report.firstCrd, ": got ", report.firstGot,
           " want ", report.firstWant, " (rtol ", rtol, ", abstol ", abstol, ")"
      raise newException(AssertionDefect, "[ceramic] allClose assertion failed")
