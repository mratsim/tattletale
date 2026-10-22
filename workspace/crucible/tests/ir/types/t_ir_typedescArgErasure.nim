## typedesc argument erasure (pruneCompileTimeParams)
##
## Test that the frontend carries type-as-value call args 1:1 and pruneCompileTimeParams drops the sentinel param plus gpuDiscard
##
## - genuine value symbols are never erased
## - non-typedesc args keep their positions through the prune
## - CUDA emission shows no `T` arg and no typedesc param
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/types/t_ir_typedescArgErasure.nim

import std/macros
import std/sequtils
import std/strutils
import std/tables
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering {.all.}
import workspace/crucible/src/codegen/passes/pass_registry

# ── Minimal stand-ins for the ceramic shapes under test ──────────────────
type
  Layout = object
    rows: int32
  TensorView[T] = object
    layout: Layout

proc make_tensor*[T](_: typedesc[T], L: Layout): TensorView[T] =
  TensorView[T](layout: L)

func make_tensor_like*[T](t: TensorView[T]): TensorView[T] =
  ## tensor_datatypes.nim passes the generic type param `T` as a typedesc
  ## VALUE argument to make_tensor.
  make_tensor(T, t.layout)

# ── Helpers ──────────────────────────────────────────────────────────────
macro toGpuAstWithInstsPruned(body: typed): untyped =
  ## Like `toGpuAstWithInsts` plus a pruneCompileTimeParams run, so tests can
  ## inspect the post-prune callee signature.
  var ctx = GpuContext()
  var typeReg = TypeRegistry(types: ctx.types)
  var gpuAst = nim_to_gpu.toGpuAst(ctx, typeReg, body)
  ctx.types = typeReg.types
  var gen: seq[GpuAst]
  for k, v in ctx.genericInsts:
    gen.add v
  var reg = PassRegistry.new()
  reg.registerPruneCompileTimeParams()
  ctx.runPasses(reg)
  result = nnkTupleConstr.newTree(newLit(gpuAst), newLit(gen))


macro toGpuAstWithInsts(body: typed): untyped =
  ## Runs the frontend like `toGpuAst` and also returns the generic instantiations, so tests can inspect the callee signatures
  var ctx = GpuContext()
  var typeReg = TypeRegistry(types: ctx.types)
  var gpuAst = nim_to_gpu.toGpuAst(ctx, typeReg, body)
  ctx.types = typeReg.types
  var gen: seq[GpuAst]
  for k, v in ctx.genericInsts:
    gen.add v
  result = nnkTupleConstr.newTree(newLit(gpuAst), newLit(gen))

proc getProcNamed(ir: GpuAst, name: string): GpuAst =
  ## Extract the gpuProc whose pName starts with `name` from a toGpuAst block.
  doAssert ir.kind == gpuBlock, "Expected gpuBlock at top level, got " & $ir.kind
  for st in ir.statements:
    if st.kind == gpuProc and st.pName.ident().startsWith(name):
      return st
  doAssert false, "No gpuProc starting with " & name & " in block"

proc findCall(n: GpuAst, name: string): GpuAst =
  ## Find the first gpuCall whose cName starts with `name` in a subtree.
  if n == nil: return nil
  if n.kind == gpuCall and n.cName.ident().startsWith(name):
    return n
  for ch in n.items:
    let r = findCall(ch, name)
    if not r.isNil: return r
  return nil

proc findInst(gen: seq[GpuAst], name: string): GpuAst =
  ## Find the generic instantiation whose pName starts with `name`.
  for p in gen:
    if p.pName.ident().startsWith(name):
      return p
  doAssert false, "No generic instantiation starting with " & name

proc paramNames(fn: GpuAst): string =
  ## Comma-joined param names of a gpuProc for assertion messages.
  var s: seq[string]
  for p in fn.pParams:
    s.add p.ident.ident()
  result = s.join(",")

proc argRepr(a: GpuAst): string =
  ## Short human-readable form of a call argument for assertion messages.
  case a.kind
  of gpuLit: "lit:" & a.lValue
  of gpuIdent: "ident:" & a.ident()
  else: $a.kind

# ── Test 1: typedesc arg (first position) erased; callee param dropped ────
block:
  let (ir, gen) = toGpuAstWithInsts:
    proc fill(_: typedesc[float32], a: int32, L: Layout) {.device.} =
      discard
    proc kernel(x: int32) {.device.} =
      var L: Layout
      fill(float32, 7, L)
  let inst = gen.findInst("fill")
  doAssert inst.pParams.len == 3,
    "frontend keeps arity 1:1, got " & $inst.pParams.len & " params: " & paramNames(inst)
  doAssert inst.pParams[0].typ.kind == gtCompileTimeValue,
    "typedesc param carries the gtCompileTimeValue sentinel"
  let call = ir.getProcNamed("kernel").findCall("fill")
  doAssert not call.isNil, "Expected a gpuCall to fill in the kernel body"
  doAssert call.cArgs.len == 3,
    "frontend keeps call arity 1:1, got " & $call.cArgs.len & " args"
  doAssert call.cArgs[0].kind == gpuDiscard,
    "typedesc arg becomes a gpuDiscard placeholder"
  doAssert call.cArgs[1].kind == gpuLit and call.cArgs[1].lValue == "7",
    "second arg must be literal 7, got " & argRepr(call.cArgs[1])
  doAssert call.cArgs[2].ident() == "L",
    "third arg must be L, got " & argRepr(call.cArgs[2])
  let (ir2, gen2) = toGpuAstWithInstsPruned:
    proc fill(_: typedesc[float32], a: int32, L: Layout) {.device.} =
      discard
    proc kernel(x: int32) {.device.} =
      var L: Layout
      fill(float32, 7, L)
  let inst2 = gen2.findInst("fill")
  doAssert inst2.pParams.len == 2,
    "prune drops the typedesc param, got " & $inst2.pParams.len &
    " params: " & paramNames(inst2)
  doAssert inst2.pParams[0].ident.ident() == "a",
    "first kept callee param must be 'a', got '" & inst2.pParams[0].ident.ident() & "'"
  doAssert inst2.pParams[1].ident.ident() == "L",
    "second kept callee param must be 'L', got '" & inst2.pParams[1].ident.ident() & "'"
  let call2 = ir2.getProcNamed("kernel").findCall("fill")
  doAssert not call2.isNil, "Expected a gpuCall to fill in the pruned kernel body"
  doAssert call2.cArgs.len == 2,
    "prune erases the typedesc arg, got " & $call2.cArgs.len & " args"
  doAssert call2.cArgs[0].kind == gpuLit and call2.cArgs[0].lValue == "7",
    "first kept arg must be literal 7, got " & argRepr(call2.cArgs[0])
  doAssert call2.cArgs[1].ident() == "L",
    "second kept arg must be L, got " & argRepr(call2.cArgs[1])
echo "  OK - typedesc arg (first position): frontend 1:1, prune drops param and arg"

# ── Test 2: typedesc arg NOT first — non-typedesc args keep position/order ──
block:
  let (ir, gen) = toGpuAstWithInsts:
    proc fill2(a: int32, _: typedesc[float32], L: Layout) {.device.} =
      discard
    proc kernel2() {.device.} =
      var L: Layout
      fill2(7, float32, L)
  let inst = gen.findInst("fill2")
  doAssert inst.pParams.len == 3,
    "frontend keeps arity 1:1 (middle position), got " & $inst.pParams.len &
    " params: " & paramNames(inst)
  doAssert inst.pParams[1].typ.kind == gtCompileTimeValue,
    "middle typedesc param carries the sentinel"
  let call = ir.getProcNamed("kernel2").findCall("fill2")
  doAssert call.cArgs[1].kind == gpuDiscard,
    "middle typedesc arg becomes a gpuDiscard placeholder"
  let (ir2, gen2) = toGpuAstWithInstsPruned:
    proc fill2(a: int32, _: typedesc[float32], L: Layout) {.device.} =
      discard
    proc kernel2() {.device.} =
      var L: Layout
      fill2(7, float32, L)
  let inst2 = gen2.findInst("fill2")
  doAssert inst2.pParams.len == 2,
    "prune drops the middle typedesc param, got " & $inst2.pParams.len &
    " params: " & paramNames(inst2)
  doAssert inst2.pParams[0].ident.ident() == "a",
    "first kept callee param must be 'a', got '" & inst2.pParams[0].ident.ident() & "'"
  doAssert inst2.pParams[1].ident.ident() == "L",
    "second kept callee param must be 'L', got '" & inst2.pParams[1].ident.ident() & "'"
  let call2 = ir2.getProcNamed("kernel2").findCall("fill2")
  doAssert not call2.isNil, "Expected a gpuCall to fill2 in the pruned kernel body"
  doAssert call2.cArgs.len == 2,
    "prune erases the middle typedesc arg, got " & $call2.cArgs.len & " args"
  doAssert call2.cArgs[0].kind == gpuLit and call2.cArgs[0].lValue == "7",
    "kept arg 0 must be literal 7 (order preserved), got " & argRepr(call2.cArgs[0])
  doAssert call2.cArgs[1].ident() == "L",
    "kept arg 1 must be L (order preserved), got " & argRepr(call2.cArgs[1])
echo "  OK - typedesc arg (middle position): order preserved through prune"

# ── Test 3: genuine value symbols are NEVER erased (INV-C3) ──────────────
block:
  let (ir, gen) = toGpuAstWithInsts:
    proc usev(a: int32, b: float32) {.device.} =
      discard
    proc kernel3() {.device.} =
      const K = 5
      usev(K, 2.5)
  let call = ir.getProcNamed("kernel3").findCall("usev")
  doAssert not call.isNil, "Expected a gpuCall to usev in the kernel body"
  doAssert call.cArgs.len == 2,
    "value args (const K, literal) must NOT be erased, got " & $call.cArgs.len & " args"
  doAssert call.cArgs[0].kind == gpuLit and call.cArgs[0].lValue == "5",
    "const value arg must be kept as literal 5, got " & argRepr(call.cArgs[0])
echo "  OK — genuine value args are never erased"

# ── Test 4: generic make_tensor_like → make_tensor(T, L) end-to-end ──────
# Exercises the real Class-C shape (tensor_datatypes.nim:47-48): the generic
# type param T passed as a typedesc value must vanish from both the emitted
# call and the emitted callee signature (REQ-C1 + REQ-C2).
block:
  let (ir, gen) = toGpuAstWithInsts:
    proc kernel4() {.device.} =
      var tv: TensorView[float32]
      let s = make_tensor_like(tv)
      discard s
  let inst = gen.findInst("make_tensor")
  doAssert inst.pParams.len == 2,
    "frontend keeps make_tensor arity 1:1, got " & $inst.pParams.len &
    " params: " & paramNames(inst)
  doAssert inst.pParams[0].typ.kind == gtCompileTimeValue,
    "typedesc param carries the sentinel"
  let call = ir.getProcNamed("kernel4").findCall("make_tensor_like")
  doAssert not call.isNil, "Expected a gpuCall to make_tensor_like in the kernel body"
  doAssert call.cArgs.len == 1,
    "make_tensor_like has no typedesc param: the tensor arg stays, got " &
    $call.cArgs.len & " args"
  let (ir2, gen2) = toGpuAstWithInstsPruned:
    proc kernel4() {.device.} =
      var tv: TensorView[float32]
      let s = make_tensor_like(tv)
      discard s
  let inst2 = gen2.findInst("make_tensor")
  doAssert inst2.pParams.len == 1,
    "prune drops the typedesc param, got " & $inst2.pParams.len &
    " params: " & paramNames(inst2)
  doAssert inst2.pParams[0].ident.ident() == "L",
    "make_tensor callee param must be 'L', got '" & inst2.pParams[0].ident.ident() & "'"
echo "  OK — generic make_tensor_like: callee signature has no typedesc param"

# ── Test 5: full CUDA emission of the make_tensor_like shape (REQ-C3 text) ──
block:
  let cu = cuda:
    proc kernel5() {.global.} =
      var tv: TensorView[float32]
      let s = make_tensor_like(tv)
      discard s
  doAssert "make_tensor" in cu, "make_tensor callee must be emitted"
  doAssert "Layout" in cu, "the layout argument must still be emitted"
  doAssert not cu.contains("float underscore"),
    "typedesc param must not appear in emitted CUDA signature, got:\n" & cu
  doAssert not cu.contains("(T,"),
    "typedesc arg T must not appear in the emitted call, got:\n" & cu
  # make_tensor must be declared and defined with exactly the Layout param
  let defLine = cu.splitLines().filterIt("make_tensor" in it and "Layout" in it and "{" in it)
  doAssert defLine.len >= 1, "expected make_tensor(Layout L) definition, got:\n" & cu
  doAssert defLine[0].contains("(Layout L)"),
    "make_tensor signature must be (Layout L), got: " & defLine[0]
echo "  OK — emitted CUDA has no T arg and no typedesc param"

echo ""
echo "  All typedesc-arg erasure tests passed."
