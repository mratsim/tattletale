## CIRv2 module verification and the debug renderer
##
## `verify` walks every proc of a module and names the module and proc in errors.
## `verifyAfter` names the producing pass.
## The renderer emits the Leng vocabulary one node per line.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/cirv2 --nimcache:nimcache/tests/cirv2 \
##     workspace/crucible/tests/cirv2/t_cirv2_module_pretty.nim

import std/strutils
import
  workspace/crucible/src/codegen/ir/cirv2_lang,
  workspace/crucible/src/codegen/ir/cirv2_build,
  workspace/crucible/src/codegen/ir/cirv2_verify,
  workspace/crucible/src/codegen/ir/cirv2_pretty

static:
  let
    f32 = tFloat(32)
    u32t = tUint(32)
    i32t = tInt(32)
    devF32 = tAptr(caDevice, f32)
    x = sym("x", devF32)

  proc goodProc(): CirProc =
    procD("scale", @[param("x", devF32)], tVoid(), @[
      varD("i", u32t, intLit(0, u32t)),
      asgnS(patE(x, sym("i", u32t)),
        mulE(f32, sym("a", f32), patE(x, sym("i", u32t)))),
    ])

  proc badProc(): CirProc =
    procD("broken", @[], tVoid(), @[
      asgnS(sym("y", f32), addE(i32t, sym("p", i32t), sym("q", i32t))),
    ])

  # module-level verify walks every proc, a violation names module and proc
  block:
    let m = moduleD("kernels", @[tStruct("pair", @[fld("lo", f32), fld("hi", f32)])],
      @[goodProc(), badProc()])
    let msg = verifyMessage(m)
    doAssert "kernels/proc[1]" in msg, msg
    doAssert "right-hand side type differs" in msg, msg
    doAssert verifyMessage(moduleD("kernels", @[], @[goodProc()])).len == 0

  # a module of clean procs verifies
  block:
    let m = moduleD("kernels", @[], @[goodProc()])
    verify(m)

  # verifyAfter names the producing pass
  block:
    let msg = verifyMessage(badProc(), "my_legalization")
    doAssert msg.startsWith("[pass my_legalization] cirv2 verify:"), msg

  # renderer: one node per line, Leng vocabulary, module with nominal type
  block:
    let m = moduleD("kernels",
      @[tStruct("pair", @[fld("lo", f32), fld("hi", f32)]), tVec(4, f32)],
      @[goodProc()])
    let text = renderModule(m)
    doAssert "(module kernels)" in text
    doAssert "(type :pair)" in text
    doAssert "(type (vector 4 (f +32)))" in text
    doAssert "(proc scale (param x (aptr device (f +32))) (void)" in text
    doAssert "(asgn (pat x i) (mul (f +32) a (pat x i)))" in text
    # one node per line: a renderer line starts with spaces or with an opening or closing paren
    for line in text.splitLines():
      doAssert line.len == 0 or line[0] == ' ' or line[0] == '(' or
        line[0] == ')', "dumper line shape: " & line
