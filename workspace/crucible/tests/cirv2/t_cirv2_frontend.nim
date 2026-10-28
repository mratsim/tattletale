## CIRv2 frontend: typed NimNode lowering to the verified normal form
##
## Test that the frontend lowers kernel procs to the normal form with their names,
## hoisted call bindings in front of their statements, memory-classed locals,
## and refusals naming unsupported constructs through the pure refusal channel:
##
## - positives lower and verify clean inside `lowerTrace`
## - negatives assert on the `lowerErr` refusal string
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/cirv2 --nimcache:nimcache/tests/cirv2 \
##     workspace/crucible/tests/cirv2/t_cirv2_frontend.nim

import std/strutils
import std/macros
import
  workspace/crucible/src/builtins/builtins_pragmas,
  workspace/crucible/src/codegen/frontend/nim_to_crucible,
  workspace/crucible/tests/cirv2/cirv2_lower_macros

proc axpyKernel(y, x: ptr UncheckedArray[float32], alpha: float32) {.global.} =
  let n = 128'i32
  for i in countup(0'i32, n - 1, 1'i32):
    y[i] = alpha * x[i] + y[i]

const axpyTrace = lowerTrace:
  axpyKernel

static:
  # kernel proc keeps its name (a nameless `(proc  ...)` is the defect)
  doAssert "(proc axpyKernel " in axpyTrace
  # while/for lower to a conditionless loop with the exit label after it
  doAssert "(loop" in axpyTrace
  doAssert "(lab lx" in axpyTrace
  # inclusive countup guard is a `le` comparison
  doAssert "(ite (le (i +32)" in axpyTrace

proc helper(x: ptr UncheckedArray[float32], i: int): float32 =
  result = x[i] * 2.0'f32

proc hoistKernel(y, x: ptr UncheckedArray[float32], alpha: float32) {.global.} =
  for i in countup(0'i32, 127'i32, 32'i32):
    y[i] = alpha * x[i] + helper(x, i)

const hoistTrace = lowerTrace:
  hoistKernel

static:
  # call binds to a temp hoisted in front of the statement that uses
  # it. The temp's `asgn` precedes the indexing assignment and never leaks
  # into a later statement
  let callIdx = hoistTrace.find("(asgn tmp (call helper_")
  let useIdx = hoistTrace.find("(asgn (pat y i)")
  doAssert callIdx >= 0
  doAssert useIdx > callIdx

const wrapperTrace = lowerTrace:
  proc axpyWrap(y: ptr UncheckedArray[float32], x: ptr UncheckedArray[float32], alpha: float32) {.global.} =
    axpyKernel(y, x, 2.5'f32)

static:
  # launchDeviceFn-shaped wrapper: its kernel call statement reaches
  # this module as a `call` to the harvested kernel proc
  doAssert "(proc axpyWrap " in wrapperTrace
  doAssert "(call axpyKernel_" in wrapperTrace

proc smemKernel(y: ptr UncheckedArray[float32]) {.global.} =
  var tile {.smem.}: array[16, float32]
  var weight {.const_mem.}: array[4, float32]
  var scale = 1.5'f32
  tile[0] = y[0]
  y[0] = tile[0] + weight[0] * scale

const smemTrace = lowerTrace:
  smemKernel

static:
  # memory-classed locals: the pragma list decides where each declaration lives,
  # unannotated locals stay in per-thread registers
  doAssert "(var threadgroup tile (array (f +32) 16) .)" in smemTrace
  doAssert "(var constant weight (array (f +32) 4) .)" in smemTrace
  doAssert "(var scale (f +32) 1.5)" in smemTrace

proc caseKernel(y: ptr UncheckedArray[int32]) {.global.} =
  var i = 0'i32
  case i
  of 1: y[0] = 1'i32
  else: y[0] = 2'i32

proc blockKernel(y: ptr UncheckedArray[int32]) {.global.} =
  block found:
    break found

proc countdownKernel(y: ptr UncheckedArray[int32]) {.global.} =
  for i in countdown(3'i32, 0'i32):
    y[i] = 0'i32

proc strKernel(y: ptr UncheckedArray[int32]) {.global.} =
  let s = "hi"
  discard s

const caseErr = lowerErr: caseKernel
const blockErr = lowerErr: blockKernel
const countdownErr = lowerErr: countdownKernel
const strErr = lowerErr: strKernel

static:
  # constructs outside the supported closure refuse by name
  doAssert "case statement" in caseErr
  doAssert "labeled block" in blockErr
  doAssert "countdown" in countdownErr
  doAssert "type not supported: string" in strErr

proc magicCallKernel(y: ptr UncheckedArray[int32], n: int32) {.global.} =
  for i in countup(0'i32, n - 1, 1'i32):
    let a = 2'i32 * i + 1'i32
    # a template-expanded infix reaches the typed tree in backtick call form,
    # a call node whose callee is the system magic
    y[i] = `mod`(a, 4'i32) + `*`(i, 2'i32)

const magicCallTrace = lowerTrace:
  magicCallKernel

static:
  # integer magics (`mod`, `*`, `+`) in call position
  # lower to the primitive operations, never to a call of a harvested empty proc
  doAssert "(mod (i +32)" in magicCallTrace
  doAssert "(mul (i +32)" in magicCallTrace
  doAssert "call mod_" notin magicCallTrace
  doAssert "call __" notin magicCallTrace

macro genShadowedRead(): untyped =
  ## Expansion shape of the ceramic layout algebra: two macro-generated lets share one name across nested blocks.
  ## The outer let is read by field after the inner one has been declared.
  let outer = genSym(nskLet, "composeMode0")
  let inner = genSym(nskLet, "composeMode0")
  result = quote do:
    block:
      let `outer` = (11'i32, 12'i32, 13'i32)
      block:
        let `inner` = (21'i32, 22'i32)
        discard `inner`
      `outer`[2]

proc shadowKernel(o: ptr UncheckedArray[int32]) {.global.} =
  o[0] = genShadowedRead()

const shadowTrace = lowerTrace:
  shadowKernel

static:
  # a reference resolves the symbol it is bound to, never the most recent
  # same-named declaration: the read is the outer let's third field
  doAssert "(dot composeMode0 F2 0)" in shadowTrace

macro genDeadLabelBlock(): untyped =
  ## A macro-generated labeled block that no break statement targets.
  ## This is the expansion shape of the `unrolled` macro in the ceramic tile kernels.
  let lbl = genSym(nskLabel, "unrolledIter")
  result = quote do:
    block:
      var acc = 0'i32
      block `lbl`:
        acc = acc + 1'i32
      acc

proc deadLabelKernel(o: ptr UncheckedArray[int32]) {.global.} =
  o[0] = genDeadLabelBlock()

const deadLabelTrace = lowerTrace:
  deadLabelKernel

static:
  # a label with no `break label` targeting it is dead and the block lowers as its body.
  # accumulator assignment reaches the kernel statements.
  # var is a macro gensym, its suffix depends on the gensym counter.
  doAssert "(asgn acc_gensym" in deadLabelTrace
  doAssert "(add (i +32)" in deadLabelTrace

macro genLabeledBreakBlock(): untyped =
  ## A macro-generated labeled block that a labeled break targets.
  ## This construct is outside the supported closure.
  let lbl = genSym(nskLabel, "found")
  result = quote do:
    block:
      block `lbl`:
        break `lbl`
      0'i32

proc labeledBreakKernel(o: ptr UncheckedArray[int32]) {.global.} =
  o[0] = genLabeledBreakBlock()

const labeledBreakErr = lowerErr: labeledBreakKernel

static:
  # a labeled break targeting the block still refuses loudly
  doAssert "labeled block" in labeledBreakErr

proc asmKernel(o: ptr UncheckedArray[int32]) {.global.} =
  var acc = 7'i32
  # a Nim `asm` statement over backtick symbols, the simdgroup-MMA expansion
  # shape of the ceramic gemm atom
  asm """
    `acc` = `acc` + 1;
  """
  o[0] = acc

const asmTrace = lowerTrace:
  asmKernel

static:
  # asm statement lowers with its template and its typed symbol operands
  doAssert "(asm" in asmTrace
  doAssert "acc" in asmTrace

type OpSel* = enum
  osA
  osB

func pick*[T](v: T, op: static OpSel): T {.inline.} =
  ## A static enum parameter selects at instantiation, the bound value
  ## shows up as the formal's instantiated type.
  when op == osA:
    result = v + 1'i32
  else:
    result = v + 2'i32

proc enumArgKernel(o: ptr UncheckedArray[int32]) {.global.} =
  o[0] = pick(10'i32, osA)
  o[1] = pick(10'i32, osB)

const enumArgTrace = lowerTrace:
  enumArgKernel

static:
  # static enum argument erases so no enum member reaches the IR
  # two instantiations are two distinct procs, one call each
  doAssert "osA" notin enumArgTrace
  doAssert "osB" notin enumArgTrace
  doAssert enumArgTrace.count("(proc pick_") == 2
  doAssert enumArgTrace.count("(call pick_") == 2
  doAssert "(param v (i +32)) (i +32)" in enumArgTrace
