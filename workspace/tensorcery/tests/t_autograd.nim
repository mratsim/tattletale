# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Autograd tape on CPU: backward walk, gradcheck, tape lifetime.
##
## Test that backward() on CPU matches f64 host central finite differences for add, tensor product and scalar multiply.
## Test that backward works before eval and after eval.
##
## Test that a fan-out tensor accumulates the sum of its consumers' contributions.
## Test that ops whose inputs all lack the grad flag produce no grad entries.
## Test that the record persists across eval and resetTape clears it at the step boundary.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_autograd.nim

import std/math

import
  workspace/tensorcery,
  workspace/tensorcery/src/backends/cpu

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
  ## Gradcheck of one op graph. backward() walks the record, one eval drains it.
  ## Each leaf gradient compares against f64 central differences of the host reference `f`.
  let ctx = Ctx()
  let a = toGradTensor(ctx, adata, n)
  let b = toGradTensor(ctx, bdata, n)
  let outT = build(ctx, a, b)
  backward(outT)
  ctx.eval()

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

    # exact on CPU: the vjp is the ones seed times the sibling, 1.0*x is exact in f32
    let ctx = Ctx()
    let a = toGradTensor(ctx, adata, n)
    let b = toGradTensor(ctx, bdata, n)
    backward(a * b)
    ctx.eval()
    doAssert toSeq[float32](grad(a)) == bdata, "mul d/da must equal b exactly"
    doAssert toSeq[float32](grad(b)) == adata, "mul d/db must equal a exactly"

  block gradcheckMulScalar:
    const s = 0.5
    let f = proc(x, y: seq[float64]): float64 = sumF64(x) * s
    let build = proc(ctx: Ctx, a, b: Tensor): Tensor = a * s
    gradcheck(n, adata, bdata, build, f, false, "mulScalar")

  block fanoutAccumulation:
    # out = (a + b) * a: a is consumed by two ops, its grad is 2a + b
    let ctx = Ctx()
    let a = toGradTensor(ctx, adata, n)
    let b = toGradTensor(ctx, bdata, n)
    let m = a + b
    let outT = m * a
    backward(outT)
    ctx.eval()

    var expectA, expectB: seq[float64]
    for i in 0 ..< n:
      expectA.add 2.0 * float64(adata[i]) + float64(bdata[i])
      expectB.add float64(adata[i])
    checkBand(toSeq[float32](grad(a)), expectA, "fanout d/da")
    checkBand(toSeq[float32](grad(b)), expectB, "fanout d/db")

    # finite differences over the same graph, f64 host math
    let f = proc(x, y: seq[float64]): float64 =
      var total = 0.0
      for i in 0 ..< x.len:
        total += (x[i] + y[i]) * x[i]
      total
    var a64, b64: seq[float64]
    for i in 0 ..< n:
      a64.add float64(adata[i])
      b64.add float64(bdata[i])
    for which in 0 .. 1:
      var expected: seq[float64]
      for i in 0 ..< n:
        expected.add centralDiff(f, a64, b64, which, i)
      let got = if which == 0: toSeq[float32](grad(a)) else: toSeq[float32](grad(b))
      checkBand(got, expected, "fanout finite differences, input " & $which)

  block evalThenBackward:
    # forward, eval, backward, eval: the walk reads the drained forward entries
    let ctx = Ctx()
    let a = toGradTensor(ctx, adata, n)
    let b = toGradTensor(ctx, bdata, n)
    let outT = a * b
    ctx.eval()
    doAssert ctx.getPendingCount() == 0
    doAssert ctx.record.len == 1, "the record persists across eval"
    backward(outT)
    doAssert ctx.getPendingCount() == 3, "seed and both vjps wait after backward"
    ctx.eval()
    doAssert toSeq[float32](grad(a)) == bdata, "eval-then-backward d/da"
    doAssert toSeq[float32](grad(b)) == adata, "eval-then-backward d/db"

  block nonGradSkip:
    # z = x + y over non-grad inputs feeds w = z * a: only a's chain produces grad entries
    let ctx = Ctx()
    let x = toTensor(ctx, adata, n)
    let y = toTensor(ctx, bdata, n)
    let a = toGradTensor(ctx, adata, n)
    doAssert not (x + y).getRequiresGrad(), "add of non-grad operands needs no grad"
    doAssert (a * 2.0).getRequiresGrad(), "scalar multiply propagates the grad flag"
    doAssert (a + x).getRequiresGrad(), "add propagates the grad flag from one side"
    let z = x + y
    let w = z * a
    backward(w)
    var gradEntries = 0
    for e in ctx.record:
      if e.isGrad:
        gradEntries.inc
    doAssert gradEntries == 2, "seed plus a's vjp only, got " & $gradEntries
    ctx.eval()
    var expected: seq[float64]
    for i in 0 ..< n:
      expected.add float64(adata[i]) + float64(bdata[i])
    checkBand(toSeq[float32](grad(a)), expected, "grad through non-grad intermediates")
    var refused = false
    try:
      discard grad(x)
    except AssertionDefect:
      refused = true
    doAssert refused, "a non-grad tensor has no gradient"

  block tapeReset:
    let ctx = Ctx()
    let a = toGradTensor(ctx, adata, n)
    let b = toGradTensor(ctx, bdata, n)
    let outT = a * b
    backward(outT)
    ctx.eval()
    doAssert toSeq[float32](grad(a)) == bdata

    resetTape(ctx)
    doAssert ctx.record.len == 0, "reset clears the record"
    doAssert ctx.getPendingCount() == 0
    var refused = false
    try:
      discard grad(a)
    except AssertionDefect:
      refused = true
    doAssert refused, "gradients die with the tape"

    # a fresh step still works and the ids never collide
    let c = toGradTensor(ctx, adata, n)
    let out2 = c * 2.0
    backward(out2)
    ctx.eval()
    var expected: seq[float64]
    for i in 0 ..< n:
      expected.add 2.0
    checkBand(toSeq[float32](grad(c)), expected, "step after reset")

  stdout.writeLine "t_autograd: OK"

main()
