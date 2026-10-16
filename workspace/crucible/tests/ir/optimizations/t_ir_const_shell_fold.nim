## Constant-shell elimination pass.
##
## Zero-arg constant-returning functions are dissolved at their call sites, and the dead definitions are deleted.
##
## Nim's semcheck leaves a zero-arg (or all-args-unused) function whose body is a single constant.
## The emitter keeps the shell and the call alive, so the pass substitutes the constant and deletes the definition.
##
## Locked:
##
## - in the IR, a zero-arg constant-returning function is gone after the pipeline, no definition and no call left
## - in the IR, a dynamic (parameter-consuming) function shell survives
## - in the IR, a shell with an effectful initializer or a foreign write survives
## - in Metal emission, constants replace the calls, scalar and struct constructor alike, with no shell emitted
##
## The IR checks run inside a `runAndSummarize` macro (the function tables live on the compiler-run GpuContext, never reaching runtime).
##
## Run:
##   nim cpp -r --hints:off --warnings:off workspace/crucible/tests/ir/optimizations/test_ir_const_shell_fold.nim (from tattletale)

import std/[macros, sequtils, strutils, tables]

import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/nim_to_gpu
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/pass_datatypes

func constFolded[S: static int](): int32 = S
func dynShift(x: int32): int32 = x + 1

proc collectCalls(n: GpuAst; calls: var seq[string]) =
  ## Appends the iSym of every called function in a subtree.
  if n.isNil:
    return
  if n.kind == gpuCall and not n.cName.isNil and not n.cName.symbol.isNil:
    calls.add n.cName.symbol.iSym
  for ch in n.items:
    collectCalls(ch, calls)

macro runAndSummarize(body: typed): string =
  ## Runs the common pass pipeline over the body exactly as a backend
  ## macro would (shared GpuContext), then returns a summary: one line per
  ## function definition in the function tables, one line per call.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  var summary = ""
  for key, fn in pairs(ctx.allFnTab):
    if fn.kind == gpuProc:
      summary.add "FN " & fn.pName.symbol.iSym &
        " params=" & $fn.pParams.len &
        " ret=" & (if fn.pRetType.isNil: "nil" else: $fn.pRetType.kind) & "\n"
  for key, fn in pairs(ctx.genericInsts):
    if fn.kind == gpuProc and not (fn.pName in ctx.allFnTab):
      summary.add "FN " & fn.pName.symbol.iSym & " (genericInsts only)\n"
  var calls: seq[string]
  for key, fn in pairs(ctx.allFnTab):
    if fn.kind == gpuProc:
      collectCalls(fn.pBody, calls)
  for c in calls:
    summary.add "CALL " & c & "\n"
  result = newLit(summary)

# ── 1. IR: the zero-arg constant shell is dissolved ──
static:
  let summary = runAndSummarize:
    proc shellKernel(C: ptr UncheckedArray[int32]) {.global.} =
      C[0] = int32(constFolded[16]() + 5)

  doAssert not summary.contains("constFolded"),
    "the constant shell and its call must both be gone, got:\n" & summary

# ── 2. IR: a dynamic function's shell survives ──
static:
  let summary = runAndSummarize:
    proc dynKernel(C: ptr UncheckedArray[int32]; x: int32) {.global.} =
      C[0] = dynShift(x)

  doAssert summary.count("FN dynShift") == 1,
    "the dynamic function definition must survive, got:\n" & summary
  doAssert summary.count("CALL dynShift") == 1,
    "the dynamic call must survive, got:\n" & summary

# ── 3. Emission (Metal): constant at the call site, no shell; dynamic survives ──
const mslFolded = metal:
  proc anchorKernel(C: ptr UncheckedArray[int32]) {.global.} =
    C[0] = int32(constFolded[16]() + 5)

const mslDyn = metal:
  proc dynAnchorKernel(C: ptr UncheckedArray[int32]; x: int32) {.global.} =
    C[0] = dynShift(x)

block:
  doAssert not mslFolded.contains("constFolded"),
    "the folded shell must not be emitted, got:\n" & mslFolded
  doAssert mslFolded.contains("16"), "the constant must appear at the call site"
  doAssert mslDyn.contains("dynShift"),
    "the dynamic function's shell must survive"
  # Emitted names carry the base58 suffix. The definition and the call share it.
  doAssert mslDyn.count("dynShift___") >= 2,
    "the dynamic shell and call must both survive, got:\n" & mslDyn

# ── 4. Emission (Metal): a struct-constant constructor folds the same way ──
type Pair = object
  a: int32
  b: int32

func constPair[P1, P2: static int](): Pair = Pair(a: P1, b: P2)

const mslPair = metal:
  proc pairKernel(C: ptr UncheckedArray[int32]) {.global.} =
    let p = constPair[16, 17]()
    C[0] = p.a + p.b + 1

block:
  doAssert not mslPair.contains("constPair"),
    "the struct-constant shell must not be emitted, got:\n" & mslPair
  doAssert mslPair.contains("16") and mslPair.contains("17"),
    "the struct constants must appear at the construction site"

# ── 5. IR: an effectful initializer disqualifies the shell ──
# A binding initializer with a call is not constant. Folding the shell
# would delete the call with the definition.
var gCounter: int32 = 0

proc counterBump(): int32 =
  gCounter = gCounter + 1
  return 7

proc candidateEffectfulInit(): int32 =
  let t = counterBump()
  return 16

static:
  let summary = runAndSummarize:
    proc effectInitKernel(C: ptr UncheckedArray[int32]) {.global.} =
      C[0] = candidateEffectfulInit()

  doAssert summary.count("CALL counterBump") == 1,
    "the effectful call must survive the pass, got:\n" & summary
  doAssert summary.count("CALL candidateEffectfulInit") == 1,
    "the effectfully-initialized shell must not fold, got:\n" & summary

# ── 6. IR: an assignment outside the body's locals disqualifies the shell ──
# A write to a global or a by-ref parameter is not a binding. Folding the shell
# would delete the write with the definition.
proc candidateWritesGlobal(): int32 =
  gCounter = 5
  return 16

static:
  let summary = runAndSummarize:
    proc globalWriteKernel(C: ptr UncheckedArray[int32]) {.global.} =
      C[0] = candidateWritesGlobal()

  doAssert summary.count("CALL candidateWritesGlobal") == 1,
    "the foreign-writing shell must not fold, got:\n" & summary
  doAssert summary.count("FN candidateWritesGlobal") == 1,
    "the foreign-writing shell definition must survive, got:\n" & summary

# ── 7. IR: a folded call keeps its side-effecting arguments ──
# A folded function whose parameter the body never reads takes no
# argument value. Substituting the constant would still delete any
# argument expression, so a call with effectful arguments must survive.
proc effectReturn(): int32 =
  gCounter = gCounter + 1
  return 7

func unreadParamFn(x: int32): int32 =
  return 9

static:
  let summary = runAndSummarize:
    proc unreadArgKernel(C: ptr UncheckedArray[int32]) {.global.} =
      C[0] = unreadParamFn(effectReturn())

  doAssert summary.count("CALL effectReturn") == 1,
    "the side-effecting argument call must survive, got:\n" & summary
  doAssert summary.count("CALL unreadParamFn") == 1,
    "the shell must not fold when an argument would be lost, got:\n" & summary

echo ""
echo "  All constant-shell elimination tests passed."
