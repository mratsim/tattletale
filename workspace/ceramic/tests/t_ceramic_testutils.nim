# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## allClose walks the logical size, broadcast views compared element-wise
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/ceramic/tests/t_ceramic_testutils.nim

import std/unittest
import workspace/ceramic
import workspace/ceramic/tests/ceramic_testutils

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "allClose - element-wise walk":

    test "broadcast difference is detected, every element compared":
      # actual broadcasts row 0 over row 1, expected holds distinct rows,
      # the 2 row-1 elements differ
      var bufActual = [1.0'f32, 2, 1, 2]
      var bufExpected = [1.0'f32, 2, 3, 4]
      let actual = make_view(cast[ptr UncheckedArray[float32]](addr bufActual[0]),
                             make_layout((2, 2), (0, 1)))
      let expected = make_view(cast[ptr UncheckedArray[float32]](addr bufExpected[0]),
                               make_layout((2, 2), (2, 1)))
      let report = allClose(actual, expected, rtol = 1e-4'f64, abstol = 1e-4'f64)
      check report.bad == 2

    test "NaN and unequal infinities never match":
      var bufActual = [float32 NaN, float32 Inf, 1.0'f32, 2.0'f32]
      var bufExpected = [1.0'f32, float32 Inf, 1.0'f32, float32 -Inf]
      let actual = make_view(cast[ptr UncheckedArray[float32]](addr bufActual[0]),
                             make_layout((2, 2), (2, 1)))
      let expected = make_view(cast[ptr UncheckedArray[float32]](addr bufExpected[0]),
                               make_layout((2, 2), (2, 1)))
      let report = allClose(actual, expected, rtol = 1e-4'f64, abstol = 1e-4'f64)
      # the NaN element and the +Inf vs -Inf element mismatch
      check report.bad == 2

    test "equal broadcast views pass":
      var bufActual = [1.0'f32, 2, 1, 2]
      var bufExpected = [1.0'f32, 2, 1, 2]
      let actual = make_view(cast[ptr UncheckedArray[float32]](addr bufActual[0]),
                             make_layout((2, 2), (0, 1)))
      let expected = make_view(cast[ptr UncheckedArray[float32]](addr bufExpected[0]),
                               make_layout((2, 2), (0, 1)))
      let report = allClose(actual, expected, rtol = 1e-4'f64, abstol = 1e-4'f64)
      check report.bad == 0

when isMainModule:
  runTest()
