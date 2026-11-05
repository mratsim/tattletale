# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Autograd tape on Metal: gradcheck and the one-eval-per-step submission proof.
##
## Test that backward() on Metal matches f64 host central finite differences for add, tensor product and scalar multiply.
## Test that a forward plus backward chain drains in one eval with submission count 1.
##
## Test that eval-then-backward costs exactly one further submission.
## Test that the record persists across eval and resetTape clears it at the step boundary.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_autograd_metal.nim

import std/math

import
  workspace/tensorcery,
  workspace/tensorcery/src/backends/metal

func frand(i: int): float32 =
  ## Deterministic host input in [0, 1) from the golden-ratio low-discrepancy sequence.
  let x = float64(i) * 0.6180339887498949
  float32(x mod 1.0)

func sumF64(a: openArray[float64]): float64 =
  for x in a:
    result += x

proc checkBand(actual: seq[float32], expected: seq[float64], label: string) =
  ## Elementwise band check of device f32 gradients against the f64 finite differences.
  doAssert actual.len == expected.len, label & ": length mismatch"
  for i in 0 ..< expected.len:
    let band = 1e-3 * abs(expected[i]) + 1e-4
    doAssert abs(float64(actual[i]) - expected[i]) <= band,
        label & ": element " & $i & " got " & $actual[i] & ", expected " & $expected[i]

proc centralDiff(f: proc(a, b: seq[float64]): float64 {.closure.}, a, b: seq[float64], which, i: int): float64 =
  ## Central finite difference of `f` at element `i` of input `which`, step 1e-2.
  const h = 1e-2
  var ap = a
  var bp = b
  if which == 0:
    ap[i] += h
  else:
    bp[i] += h
  let plus = f(ap, bp)
  if which == 0:
    ap[i] -= 2 * h
  else:
    bp[i] -= 2 * h
  result = (plus - f(ap, bp)) / (2 * h)

proc gradcheck(n: int, adata, bdata: seq[float32],
    build: proc(ctx: Ctx, a, b: Tensor): Tensor {.closure.},
    f: proc(a, b: seq[float64]): float64 {.closure.}, checkSecond: bool, label: string) =
  ## Gradcheck of one op graph. backward() walks the record, one eval must submit once.
  ## Each leaf gradient compares against f64 central differences.
  let ctx = Ctx()
  let a = toGradTensor(ctx, adata, bkMetal, n)
  let b = toGradTensor(ctx, bdata, bkMetal, n)
  let outT = build(ctx, a, b)
  backward(outT)

  resetSubmissionCount()
  ctx.eval()
  doAssert submissionCount == 1, label & ": forward plus grad entries drain in ONE submission"

  var a64, b64: seq[float64]
  for i in 0 ..< n:
    a64.add float64(adata[i])
    b64.add float64(bdata[i])

  for which in 0 .. 1:
    if which == 1 and not checkSecond:
      continue
    var expected: seq[float64]
    for i in 0 ..< n:
      expected.add centralDiff(f, a64, b64, which, i)
    let got = if which == 0: toSeq[float32](grad(a)) else: toSeq[float32](grad(b))
    let side = if which == 0: " d/da" else: " d/db"
    checkBand(got, expected, label & side)

proc main() =
  let n = 4
  var adata, bdata: seq[float32]
  for i in 0 ..< n:
    adata.add frand(i + 11)
    bdata.add frand(i + 23)

  block gradcheckAdd:
    let f = proc(x, y: seq[float64]): float64 = sumF64(x) + sumF64(y)
    let build = proc(ctx: Ctx, a, b: Tensor): Tensor = a + b
    gradcheck(n, adata, bdata, build, f, true, "add")

  block gradcheckMul:
    let f = proc(x, y: seq[float64]): float64 =
      var total = 0.0
      for i in 0 ..< x.len:
        total += x[i] * y[i]
      total
    let build = proc(ctx: Ctx, a, b: Tensor): Tensor = a * b
    gradcheck(n, adata, bdata, build, f, true, "mul")

  block gradcheckMulScalar:
    const s = 0.5
    let f = proc(x, y: seq[float64]): float64 = sumF64(x) * s
    let build = proc(ctx: Ctx, a, b: Tensor): Tensor = a * s
    gradcheck(n, adata, bdata, build, f, false, "mulScalar")

  block oneEvalPerStep:
    # forward chain with fan-out plus backward, all in ONE record, ONE eval, ONE submission
    let ctx = Ctx()
    let a = toGradTensor(ctx, adata, bkMetal, n)
    let b = toGradTensor(ctx, bdata, bkMetal, n)
    let m = a + b
    let p = m * a
    let outT = p * 0.5
    backward(outT)
    doAssert ctx.getPendingCount() == 8, "3 forward and 5 grad entries wait in one record"

    resetSubmissionCount()
    ctx.eval()
    doAssert submissionCount == 1,
        "forward plus backward must drain as ONE submission, got " & $submissionCount

    # d(sum(out))/da = (2a + b) * 0.5, d/db = a * 0.5
    var expectA, expectB: seq[float64]
    for i in 0 ..< n:
      expectA.add (2.0 * float64(adata[i]) + float64(bdata[i])) * 0.5
      expectB.add float64(adata[i]) * 0.5
    checkBand(toSeq[float32](grad(a)), expectA, "oneEvalPerStep d/da")
    checkBand(toSeq[float32](grad(b)), expectB, "oneEvalPerStep d/db")

  block evalThenBackward:
    # forward, eval, backward, eval: exactly one submission per eval
    let ctx = Ctx()
    let a = toGradTensor(ctx, adata, bkMetal, n)
    let b = toGradTensor(ctx, bdata, bkMetal, n)
    let outT = a * b
    resetSubmissionCount()
    ctx.eval()
    doAssert submissionCount == 1, "the forward eval submits once"
    backward(outT)
    ctx.eval()
    doAssert submissionCount == 2, "the grad eval adds exactly one submission"

    var expectA, expectB: seq[float64]
    for i in 0 ..< n:
      expectA.add float64(bdata[i])
      expectB.add float64(adata[i])
    checkBand(toSeq[float32](grad(a)), expectA, "evalThenBackward d/da")
    checkBand(toSeq[float32](grad(b)), expectB, "evalThenBackward d/db")

  block tapeReset:
    let ctx = Ctx()
    let a = toGradTensor(ctx, adata, bkMetal, n)
    let b = toGradTensor(ctx, bdata, bkMetal, n)
    let outT = a * b
    backward(outT)
    ctx.eval()
    doAssert toSeq[float32](grad(a)) == bdata, "mul d/da before reset"

    resetTape(ctx)
    doAssert ctx.record.len == 0, "reset clears the record"
    var refused = false
    try:
      discard grad(a)
    except AssertionDefect:
      refused = true
    doAssert refused, "gradients die with the tape"

    # a fresh step still works and the ids never collide
    let c = toGradTensor(ctx, adata, bkMetal, n)
    let out2 = c * 2.0
    backward(out2)
    ctx.eval()
    var expected: seq[float64]
    for i in 0 ..< n:
      expected.add 2.0
    checkBand(toSeq[float32](grad(c)), expected, "step after reset")

  stdout.writeLine "t_autograd_metal: OK"

main()
