## NVRTC: constexpr temporaries leaking into expression slots
##
## Two bugs fixed:
##
## Bug 1: ObjConstr used as dot-parent
##   Nim constant-folds `const tup = (a, b)` so `tup[0]` becomes
##   `nnkBracketExpr(nnkTupleConstr(a, b), 0)`. Crucible converts this to
##   `gpuDot(dParent = gpuObjConstr(...), field=Field0)`.
##   The codegen emitted bare `{val}.Field0` which is invalid C++ because
##   a braced-init-list `{val}` is not an expression — you can't access
##   members on it.
##   Fix: emit `TypeName{val}.Field0` (C++ functional-style cast).
##
## Bug 2: Constexpr declarations used as expression values
##   A `const` inside the cuda: block becomes a `gpuConstexpr` node in the
##   IR. If this node ends up in an expression slot (e.g. as the RHS of a
##   `gpuVar.vInit` or as `gpuDot.dParent`), the codegen emits:
##     `Int_0 x = constexpr Type tmp = {};`
##   This is invalid because `constexpr Type tmp = {}` is a declaration
##   statement, not an expression — it can't be inlined.
##   Fix: the `liftConstexprFrom` pass recursively walks every statement's
##   expression children. When it finds a `gpuConstexpr` in an expression
##   slot, it lifts the constexpr to a preceding standalone statement and
##   replaces it with a reference to the constexpr's identifier.
##
## Patterns:
##   A — const + let in same scope — constexpr lifts to preceding stmt.
##   B — constexpr Int values in arithmetic — pass lifts before binop.
##   C — template with block { const tmp; yield tmp } — block-init lift.
##   D — let-tuple bracket access D[0] — gpuObjConstr in expression.
##   E — block with two constexprs and tuple field access — pass lift.
##   F — constexpr tuple tup[0] — gpuObjConstr as gpuDot.dParent.
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_constexpr_temp.nim

import std/[unittest, macros]
import workspace/crucible/src/codegen/nvrtc

# ── Minimal types (no ceramic deps) ──

type
  Int*[V: static int] = object
    discard

  Tuple2*[A, B] = object
    f0: A
    f1: B

template `+`*[A, B: static int](a: Int[A], b: Int[B]): Int[A + B] =
  Int[A + B]()

template `*`*[A, B: static int](a: Int[A], b: Int[B]): Int[A * B] =
  Int[A * B]()

template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V

# Pattern A: constexpr tuple used directly in let RHS (simulates what evalOnceAs produces)
const kernelA = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp
    C[0] = 1.0'f32

# Pattern B: constexpr tuple used in arithmetic expression
const kernelB = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    const a {.genSym.} = Int[8]()
    const b {.genSym.} = Int[16]()
    let x = Int[0]() + a * b
    C[0] = float32(toIntVal x)

# Pattern C: let-const-let where constexpr is used as function arg
template wrapConst(a, b: untyped): untyped =
  block:
    const tmp {.genSym.} = Tuple2[typeof(a), typeof(b)](f0: a, f1: b)
    tmp

const kernelC = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let pair = wrapConst(Int[8](), Int[16]())
    C[0] = 1.0'f32

# Pattern D: tuple bracket access on a const (like expandTuple pattern)
const kernelD = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let pos = (Int[0](), Int[0]())
    let D = (Int[1](), Int[8]())
    let idx = Int[0]() + D[0] * pos[0] + D[1] * pos[1]
    C[0] = float32(toIntVal idx)

# Pattern E: constexpr default-initialized tuple used as expression
const kernelE = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    # Simulates what evalOnceAs does for crd2idx with (0,0) coordinates
    let idx = block:
      const coord {.genSym.} = (Int[0](), Int[0]())
      const stride {.genSym.} = (Int[1](), Int[8]())
      Int[0]() + stride[0] * coord[0] + stride[1] * coord[1]
    C[0] = float32(toIntVal idx)

# Pattern F: tuple field access on constexpr tuple
const kernelF = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    const tup {.genSym.} = (Int[8](), Int[16]())
    let first = tup[0]
    C[0] = float32(toIntVal first)

suite "Crucible - constexpr tuple init in expression slots":
  test "Pattern A — constexpr tuple in let RHS (block-unwrapped)":
    var output: array[1, float32]
    var nv = initNvrtc(kernelA)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", output, ())
    check output[0] == 1.0'f32

  test "Pattern B — constexpr in arithmetic expression":
    var output: array[1, float32]
    var nv = initNvrtc(kernelB)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", output, ())
    check output[0] == 128.0'f32

  test "Pattern C — template wrapConst (block with constexpr)":
    var output: array[1, float32]
    var nv = initNvrtc(kernelC)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", output, ())
    check output[0] == 1.0'f32

  test "Pattern D — tuple bracket access on let":
    var output: array[1, float32]
    var nv = initNvrtc(kernelD)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", output, ())
    check output[0] == 0.0'f32

  test "Pattern E — block with constexpr temp (evalOnceAs sim)":
    var output: array[1, float32]
    var nv = initNvrtc(kernelE)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", output, ())
    check output[0] == 0.0'f32

  test "Pattern F — constexpr tuple field access":
    var output: array[1, float32]
    var nv = initNvrtc(kernelF)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", output, ())
    check output[0] == 8.0'f32
