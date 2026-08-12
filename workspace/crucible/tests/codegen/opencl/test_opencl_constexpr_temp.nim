## OpenCL: constexpr temporaries leaking into expression slots
##
## Two bugs fixed:
##
## Bug 1: ObjConstr used as dot-parent
##   Nim constant-folds `const tup = (a, b)` so `tup[0]` becomes
##   `nnkBracketExpr(nnkTupleConstr(a, b), 0)`. Crucible converts this to
##   `gpuDot(dParent = gpuObjConstr(...), field=Field0)`.
##   The codegen emitted bare `{val}.Field0` — invalid because braced-
##   init-lists are not expressions and can't use member access.
##   Fix (OpenCL): emit `(struct Type){val}.Field0` (C99 compound literal).
##
## Bug 2: Constexpr declarations used as expression values
##   A `const` inside the kernel block becomes a `gpuConstexpr` node. If
##   this node ends up in an expression slot (gpuVar.vInit, gpuDot.dParent,
##   etc.), the codegen emits `Type x = constexpr Type tmp = {};` which
##   is invalid — `constexpr` is a declaration, not an expression value.
##   Fix: `liftConstexprFrom` pass recursively walks expression children
##   of every statement. When it finds a `gpuConstexpr` in an expression
##   slot, it lifts it to a preceding standalone statement and replaces
##   it with a reference to the constexpr's identifier.
##
## Pattern A: const + let in same scope — constexpr lifts to preceding stmt.
## Pattern B: constexpr Int values in arithmetic — pass lifts before binop.
## Pattern C: template with block { const tmp; yield tmp } — block-init lift.
## Pattern D: let-tuple bracket access D[0] — gpuObjConstr in expression.
## Pattern E: block with two constexprs and tuple field access — pass lift.
## Pattern F: constexpr tuple tup[0] — gpuObjConstr as gpuDot.dParent.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_constexpr_temp.nim

import std/[unittest]
import workspace/crucible

type
  Int*[V: static int] = object
    discard
  Tuple2*[A, B] = object
    f0: A
    f1: B

template `+`*[A, B: static int](a: Int[A], b: Int[B]): Int[A + B] = Int[A + B]()
template `*`*[A, B: static int](a: Int[A], b: Int[B]): Int[A * B] = Int[A * B]()
template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V

# Pattern A — constexpr tuple in let RHS
const kernelA = opencl:
  proc testA(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp
    C[0] = 1'u32

# Pattern B — constexpr in arithmetic expression
const kernelB = opencl:
  proc testB(C: ptr UncheckedArray[uint32]) {.global.} =
    const a {.genSym.} = Int[8]()
    const b {.genSym.} = Int[16]()
    let x = Int[0]() + a * b
    C[0] = uint32(toIntVal x)

# Pattern C — template wrapConst (block with constexpr)
template wrapConst(a, b: untyped): untyped =
  block:
    const tmp {.genSym.} = Tuple2[typeof(a), typeof(b)](f0: a, f1: b)
    tmp
const kernelC = opencl:
  proc testC(C: ptr UncheckedArray[uint32]) {.global.} =
    let pair = wrapConst(Int[8](), Int[16]())
    C[0] = 1'u32

# Pattern D — tuple bracket access on let
const kernelD = opencl:
  proc testD(C: ptr UncheckedArray[uint32]) {.global.} =
    let pos = (Int[0](), Int[0]())
    let D = (Int[1](), Int[8]())
    let idx = Int[0]() + D[0] * pos[0] + D[1] * pos[1]
    C[0] = uint32(toIntVal idx)

# Pattern E — block with constexpr temp (evalOnceAs sim)
const kernelE = opencl:
  proc testE(C: ptr UncheckedArray[uint32]) {.global.} =
    let idx = block:
      const coord {.genSym.} = (Int[0](), Int[0]())
      const stride {.genSym.} = (Int[1](), Int[8]())
      Int[0]() + stride[0] * coord[0] + stride[1] * coord[1]
    C[0] = uint32(toIntVal idx)

# Pattern F — constexpr tuple field access (original bug)
const kernelF = opencl:
  proc testF(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (Int[8](), Int[16]())
    let first = tup[0]
    C[0] = uint32(toIntVal first)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "OpenCL - constexpr tuple init":
    test "Pattern A — constexpr tuple in let RHS":
      var engine = bkOpenCL.init()
      engine.ingest(kernelA)
      var res: array[1, uint32]
      engine.run("testA", res, ())
      check res[0] == 1

    test "Pattern B — constexpr in arithmetic":
      var engine = bkOpenCL.init()
      engine.ingest(kernelB)
      var res: array[1, uint32]
      engine.run("testB", res, ())
      check res[0] == 128

    test "Pattern C — template wrapConst":
      var engine = bkOpenCL.init()
      engine.ingest(kernelC)
      var res: array[1, uint32]
      engine.run("testC", res, ())
      check res[0] == 1

    test "Pattern D — tuple bracket access":
      var engine = bkOpenCL.init()
      engine.ingest(kernelD)
      var res: array[1, uint32]
      engine.run("testD", res, ())
      check res[0] == 0

    test "Pattern E — block with constexpr temp":
      var engine = bkOpenCL.init()
      engine.ingest(kernelE)
      var res: array[1, uint32]
      engine.run("testE", res, ())
      check res[0] == 0

    test "Pattern F — constexpr tuple field access":
      var engine = bkOpenCL.init()
      engine.ingest(kernelF)
      var res: array[1, uint32]
      engine.run("testF", res, ())
      check res[0] == 8

when isMainModule:
  runTest()
