## Vulkan (GLSL): constexpr temporaries leaking into expression slots
##
## Two bugs fixed:
##
## Bug 1: ObjConstr used as dot-parent
##   Nim constant-folds `const tup = (a, b)` so `tup[0]` becomes a
##   `gpuDot(dParent = gpuObjConstr(...), field=Field0)` in the IR.
##   The codegen emitted bare `{val}.Field0` — invalid.
##   Fix (Vulkan/GLSL): GLSL constructors TypeName(val) are valid
##   expressions, so `TypeName(val).Field0` works unchanged.
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

import std/[unittest, strformat]
import workspace/crucible/src/codegen/vk

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

# Pattern A
const kernelA = vulkan:
  proc testA(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp; C[0] = 1'u32

# Pattern B
const kernelB = vulkan:
  proc testB(C: ptr UncheckedArray[uint32]) {.global.} =
    const a {.genSym.} = Int[8](); const b {.genSym.} = Int[16]()
    let x = Int[0]() + a * b; C[0] = uint32(toIntVal x)

# Pattern C
template wrapConst(a, b: untyped): untyped =
  block:
    const tmp {.genSym.} = Tuple2[typeof(a), typeof(b)](f0: a, f1: b)
    tmp
const kernelC = vulkan:
  proc testC(C: ptr UncheckedArray[uint32]) {.global.} =
    let pair = wrapConst(Int[8](), Int[16]()); C[0] = 1'u32

# Pattern D
const kernelD = vulkan:
  proc testD(C: ptr UncheckedArray[uint32]) {.global.} =
    let pos = (Int[0](), Int[0]()); let D = (Int[1](), Int[8]())
    let idx = Int[0]() + D[0] * pos[0] + D[1] * pos[1]
    C[0] = uint32(toIntVal idx)

# Pattern E
const kernelE = vulkan:
  proc testE(C: ptr UncheckedArray[uint32]) {.global.} =
    let idx = block:
      const coord {.genSym.} = (Int[0](), Int[0]())
      const stride {.genSym.} = (Int[1](), Int[8]())
      Int[0]() + stride[0] * coord[0] + stride[1] * coord[1]
    C[0] = uint32(toIntVal idx)

# Pattern F
const kernelF = vulkan:
  proc testF(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (Int[8](), Int[16]())
    let first = tup[0]; C[0] = uint32(toIntVal first)

when isMainModule:
  echo kernelA
  echo "\n---"
  echo kernelB
  echo "\n---"
  echo kernelC
  echo "\n---"
  echo kernelD
  echo "\n---"
  echo kernelE
  echo "\n---"
  echo kernelF

suite "Vulkan - constexpr tuple init":
  test "Pattern A — constexpr tuple in let RHS":
    var ctx = initVulkan(); defer: ctx.shutdown()
    let r = execVulkan(ctx, kernelA, "testA", outputBytes = 4, inputs = [])
    check cast[ptr uint32](r[0].addr)[] == 1
  test "Pattern B — constexpr in arithmetic":
    var ctx = initVulkan(); defer: ctx.shutdown()
    let r = execVulkan(ctx, kernelB, "testB", outputBytes = 4, inputs = [])
    check cast[ptr uint32](r[0].addr)[] == 128
  test "Pattern C — template wrapConst":
    var ctx = initVulkan(); defer: ctx.shutdown()
    let r = execVulkan(ctx, kernelC, "testC", outputBytes = 4, inputs = [])
    check cast[ptr uint32](r[0].addr)[] == 1
  test "Pattern D — tuple bracket access":
    var ctx = initVulkan(); defer: ctx.shutdown()
    let r = execVulkan(ctx, kernelD, "testD", outputBytes = 4, inputs = [])
    check cast[ptr uint32](r[0].addr)[] == 0
  test "Pattern E — block with constexpr temp":
    var ctx = initVulkan(); defer: ctx.shutdown()
    let r = execVulkan(ctx, kernelE, "testE", outputBytes = 4, inputs = [])
    check cast[ptr uint32](r[0].addr)[] == 0
  test "Pattern F — constexpr tuple field access":
    var ctx = initVulkan(); defer: ctx.shutdown()
    let r = execVulkan(ctx, kernelF, "testF", outputBytes = 4, inputs = [])
    check cast[ptr uint32](r[0].addr)[] == 8
