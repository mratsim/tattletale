## CIRv2 axpy fixture: the positive normal-form kernel
##
## An axpy-shaped kernel (y = a*x + y over an aptr of float32) verifies clean and renders in the Leng vocabulary.
## The kernel uses a conditionless loop with a leading guard.
## Pointer indexing goes through `pat`, arithmetic is typed, and an `assume` states the loop bound.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/cirv2 --nimcache:nimcache/tests/cirv2 \
##     workspace/crucible/tests/cirv2/t_cirv2_axpy.nim

import std/strutils
import
  workspace/crucible/src/codegen/ir/cirv2_lang,
  workspace/crucible/src/codegen/ir/cirv2_build,
  workspace/crucible/src/codegen/ir/cirv2_verify,
  workspace/crucible/src/codegen/ir/cirv2_pretty,
  workspace/crucible/tests/cirv2/cirv2_testutils

proc axpyFixture(): CirProc =
  ## y[i] = a*x[i] + y[i] for i in 0 ..< n, as the normal form shapes it:
  ## conditionless loop, leading guard via `ite`, forward `jmp` out to a lab
  ## after the loop, `assume` stating the loop bound.
  let
    f32 = tFloat(32)
    u32t = tUint(32)
    devF32 = tAptr(caDevice, f32)
  let
    i = sym("i", u32t)
    n = sym("n", u32t)
    a = sym("a", f32)
    x = sym("x", devF32)
    y = sym("y", devF32)
  result = procD("axpy",
    @[param("n", u32t), param("a", f32), param("x", devF32),
      param("y", devF32)],
    tVoid(),
    @[
      varD("i", u32t, intLit(0, u32t)),
      loopS(@[
        assumeS(ltE(i, n)),
        iteS(ltE(i, n),
          @[
            asgnS(patE(y, i),
              addE(f32, mulE(f32, a, patE(x, i)), patE(y, i))),
            asgnS(i, addE(u32t, i, intLit(1, u32t))),
          ],
          @[jmpS("loop_exit")]),
      ]),
      labS("loop_exit"),
      retS(),
    ])

static:
  let fn = axpyFixture()
  verify(fn)
  # a second verify of the same tree is idempotent
  verify(fn)
  # renderer names the Leng vocabulary
  let text = renderProc(fn)
  doAssert "(aptr device (f +32))" in text
  doAssert "(add (f +32)" in text
  doAssert "(pat y i)" in text
  doAssert "(assume (lt (u +32) i n))" in text
  doAssert "(lab loop_exit)" in text
  doAssert "(loop" in text
