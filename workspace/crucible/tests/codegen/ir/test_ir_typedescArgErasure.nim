## Phase 0: typedesc-argument erasure test
##
## Verifies the frontend erases `typedesc`-typed arguments at gpuCall
## construction and drops the matching `typedesc` params from the emitted
## callee signature, so type-as-value calls (e.g. make_tensor_like →
## make_tensor(T, L)) stay arity-consistent. Non-typedesc args keep their
## exact position/order; genuine value symbols are never erased.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_typedescArgErasure.nim

import std/macros
import std/sequtils
import std/strutils
import std/tables
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/ir/gpu_type_constructors
import workspace/crucible/src/codegen/ir/nim_to_gpu

# ── Minimal stand-ins for the ceramic shapes under test ──────────────────
type
  Layout = object
  TensorView[T] = object
    layout: Layout

proc make_tensor*[T](_: typedesc[T]; L: Layout): TensorView[T] =
  TensorView[T](layout: L)

func make_tensor_like*[T](t: TensorView[T]): TensorView[T] =
  ## Mirrors tensor_datatypes.nim:47-48 — the generic type param `T` is passed
  ## as a typedesc VALUE argument to make_tensor.
  make_tensor(T, t.layout)

# ── Helpers ──────────────────────────────────────────────────────────────
macro toGpuAstWithInsts(body: typed): untyped =
  ## Runs the frontend like `toGpuAst` but also returns the generic
  ## instantiations (GpuGenericsInfo), so tests can inspect the emitted callee
  ## signatures. Mirrors GpuContext extraction used by the runtime codegen.
  var ctx = GpuContext()
  var typeReg = TypeRegistry(types: ctx.types)
  var gpuAst = nim_to_gpu.toGpuAst(ctx, typeReg, body)
  ctx.types = typeReg.types
  var gen = GpuGenericsInfo()
  for k, v in ctx.genericInsts:
    gen.procs.add v
  result = nnkTupleConstr.newTree(newLit(gpuAst), newLit(gen))

proc getProcNamed(ir: GpuAst; name: string): GpuAst =
  ## Extract the gpuProc whose pName starts with `name` from a toGpuAst block.
  doAssert ir.kind == gpuBlock, "Expected gpuBlock at top level, got " & $ir.kind
  for st in ir.statements:
    if st.kind == gpuProc and st.pName.ident().startsWith(name):
      return st
  doAssert false, "No gpuProc starting with " & name & " in block"

proc findCall(n: GpuAst; name: string): GpuAst =
  ## Find the first gpuCall whose cName starts with `name` in a subtree.
  if n == nil: return nil
  if n.kind == gpuCall and n.cName.ident().startsWith(name):
    return n
  for ch in n.items:
    let r = findCall(ch, name)
    if not r.isNil: return r
  return nil

proc findInst(gen: GpuGenericsInfo; name: string): GpuAst =
  ## Find the generic instantiation whose pName starts with `name`.
  for p in gen.procs:
    if p.pName.ident().startsWith(name):
      return p
  doAssert false, "No generic instantiation starting with " & name & " in GpuGenericsInfo"

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
    proc fill(_: typedesc[float32]; a: int32; L: Layout) {.device.} =
      discard
    proc kernel(x: int32) {.device.} =
      var L: Layout
      fill(float32, 7, L)
  let inst = gen.findInst("fill")
  doAssert inst.pParams.len == 2,
    "typedesc param must be dropped from the emitted callee signature, got " &
    $inst.pParams.len & " params: " & paramNames(inst)
  doAssert inst.pParams[0].ident.ident() == "a",
    "first kept callee param must be 'a', got '" & inst.pParams[0].ident.ident() & "'"
  doAssert inst.pParams[1].ident.ident() == "L",
    "second kept callee param must be 'L', got '" & inst.pParams[1].ident.ident() & "'"
  let call = ir.getProcNamed("kernel").findCall("fill")
  doAssert not call.isNil, "Expected a gpuCall to fill in the kernel body"
  doAssert call.cArgs.len == 2,
    "typedesc arg must be erased from gpuCall, got " & $call.cArgs.len & " args"
  doAssert call.cArgs[0].kind == gpuLit and call.cArgs[0].lValue == "7",
    "first kept arg must be literal 7, got " & argRepr(call.cArgs[0])
  doAssert call.cArgs[1].ident() == "L",
    "second kept arg must be L, got " & argRepr(call.cArgs[1])
echo "  OK — typedesc arg (first position) erased; matching callee param dropped"

# ── Test 2: typedesc arg NOT first — non-typedesc args keep position/order ──
block:
  let (ir, gen) = toGpuAstWithInsts:
    proc fill2(a: int32; _: typedesc[float32]; L: Layout) {.device.} =
      discard
    proc kernel2() {.device.} =
      var L: Layout
      fill2(7, float32, L)
  let inst = gen.findInst("fill2")
  doAssert inst.pParams.len == 2,
    "typedesc param (middle) must be dropped, got " & $inst.pParams.len &
    " params: " & paramNames(inst)
  doAssert inst.pParams[0].ident.ident() == "a",
    "first kept callee param must be 'a', got '" & inst.pParams[0].ident.ident() & "'"
  doAssert inst.pParams[1].ident.ident() == "L",
    "second kept callee param must be 'L', got '" & inst.pParams[1].ident.ident() & "'"
  let call = ir.getProcNamed("kernel2").findCall("fill2")
  doAssert not call.isNil, "Expected a gpuCall to fill2 in the kernel body"
  doAssert call.cArgs.len == 2,
    "typedesc arg must be erased (middle position), got " & $call.cArgs.len & " args"
  doAssert call.cArgs[0].kind == gpuLit and call.cArgs[0].lValue == "7",
    "kept arg 0 must be literal 7 (order preserved), got " & argRepr(call.cArgs[0])
  doAssert call.cArgs[1].ident() == "L",
    "kept arg 1 must be L (order preserved), got " & argRepr(call.cArgs[1])
echo "  OK — typedesc arg (middle position) erased; non-typedesc args keep order"

# ── Test 3: genuine value symbols are NEVER erased (INV-C3) ──────────────
block:
  let (ir, gen) = toGpuAstWithInsts:
    proc usev(a: int32; b: float32) {.device.} =
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
  doAssert inst.pParams.len == 1,
    "make_tensor callee must have only the Layout param, got " & $inst.pParams.len &
    " params: " & paramNames(inst)
  doAssert inst.pParams[0].ident.ident() == "L",
    "make_tensor callee param must be 'L', got '" & inst.pParams[0].ident.ident() & "'"
  let call = ir.getProcNamed("kernel4").findCall("make_tensor_like")
  doAssert not call.isNil, "Expected a gpuCall to make_tensor_like in the kernel body"
  doAssert call.cArgs.len == 1,
    "make_tensor_like must keep only the tensor arg, got " & $call.cArgs.len & " args"
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
