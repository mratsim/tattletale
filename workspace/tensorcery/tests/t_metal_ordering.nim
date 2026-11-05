# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal in-encoder ordering stress: a chained elementwise record across one eval.
##
## Test that 10 chained elementwise kernels, each consuming the previous output, on N = 2^22 f32 elements,
## compute the f64 host reference in ONE submission, and that 20 such evals in one process all stay correct.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_metal_ordering.nim

import std/math
import
  workspace/tensorcery,
  workspace/tensorcery/src/backends/metal

const
  n = 1 shl 22
  chainOps = 10
  evals = 20
  scalars = [2.0'f64, 0.5, 1.5, 0.25, 4.0, 0.5, 2.0, 0.75, 1.5, 0.5]
    ## Op with even index multiplies by its scalar. Op with odd index adds the tensor to itself, a factor of 2.

func frand(i: int): float32 =
  ## Deterministic host input in [0, 1) from the golden-ratio low-discrepancy sequence.
  let x = float64(i) * 0.6180339887498949
  float32(x mod 1.0)

proc main() =
  let ctx = Ctx()
  var x: seq[float32]
  for i in 0 ..< n:
    x.add frand(i)
  let a = toTensor(ctx, x, bkMetal, n)

  var total = 1.0'f64
  for k in 0 ..< chainOps:
    let factor = if k mod 2 == 0: scalars[k div 2] else: 2.0
    total *= factor

  for iter in 1 .. evals:
    resetSubmissionCount()
    var t = a
    for k in 0 ..< chainOps:
      t = if k mod 2 == 0: t * scalars[k div 2] else: t + t
    doAssert ctx.getPendingCount() == chainOps
    ctx.eval()
    doAssert submissionCount == 1,
        "eval " & $iter & " must be ONE submission, got " & $submissionCount

    let actual = toSeq[float32](t)
    for i in 0 ..< n:
      let expected = float64(x[i]) * total
      let band = 1e-5 * abs(expected) + 1e-6
      doAssert abs(float64(actual[i]) - expected) <= band,
          "eval " & $iter & ": element " & $i & " got " & $actual[i] &
          ", expected " & $expected

main()
