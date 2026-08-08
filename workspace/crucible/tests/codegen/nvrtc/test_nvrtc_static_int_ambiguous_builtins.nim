## NVRTC: `max` on Int[N] — ambiguous builtin name with the +/*-style overload set
##
## Int[N] is a CuTe-style compile-time integer: its value lives in the type
## parameter. The max overload set is recreated below (mirroring ceramic's
## int_tuples_datatypes.nim) so this test is self-contained — no ceramic
## dependency. Two cases:
##   - compile-time int (a static-int type or an int literal): the result is a
##     static-int type (max(Int[5](), 7) -> Int[7]); everything is handled at
##     the Nim level and the C compiler can optimize the code away.
##   - dynamic int (a runtime value): the codegen materializes the type-level
##     integer into a literal int (max(Int[5](), h) -> max(5, h)).
##
## The five cases tested:
##   1. both operands Int[N] types:  max(Int[V](), Int[U]()) -> Int[max(V,U)]() (resolved at Nim compile time)
##   2. literal on the right:        max(Int[V](), 1)       -> Int[max(V,1)]()
##   3. literal on the left:         max(1, Int[V]())       -> Int[max(1,V)]()
##   4. runtime value on the right:  max(Int[V](), h)       -> max(V, h) (plain int)
##   5. runtime value on the left:   max(h, Int[V]())       -> max(h, V) (plain int)
## Static-int results (cases 1-3) are verified via toIntVal (compile-time
## extraction from the type); dynamic results (cases 4-5) are plain ints.
##
## What this tests / detects / prevents:
## max (like min and abs) is an "ambiguous builtin": both Nim and the GPU
## languages define it, so crucible registers it name-only and forwards calls
## to the backend's native max. That is correct for basic operands — the case
## CUDA/OpenCL/Vulkan/WebGPU support natively. But the name-only forwarding
## must not swallow library-defined overloads: with an empty Int[N] operand
## (the value lives only in the type parameter) the call must resolve to the
## overloads below and compile through the same body-parsed path as +/* —
## never reach the backend's native max, which has no overload for an empty
## struct:
##     Int10 gap = max(1, Int10{});
##     no instance of overloaded function "max" matches the argument list
## This test pins that contract for all five shapes. It is currently RED on
## the compile step: ambiguous-builtin overload bodies are not parsed, so the
## Int[N] calls are forwarded and NVRTC rejects them. Once the parser accepts
## both if-expression branch forms (ElifExpr/ElseExpr and ElifBranch/Else),
## ambiguous builtins follow the same rules as +/* — their overload bodies
## parse, and this suite goes green.
##
## Run:
##   cd tattletale
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##     nim cpp -r --hints:off --warnings:off \
##       --outdir:build/tests/nvrtc --nimcache:nimcache/tests/nvrtc \
##       workspace/crucible/tests/codegen/nvrtc/test_nvrtc_static_int_ambiguous_builtins.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc

# ── static-int type + max overload set (mirrors ceramic) ────────────────
type Int*[V: static int] = object

template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V

template genBinOp(op: untyped): untyped =
  template op*[V, U: static int](a: Int[V]; b: Int[U]): auto = Int[op(V, U)]()
  func op*[V: static int](a: Int[V]; b: static int): auto {.inline.} = Int[op(V, b)]()
  func op*[V: static int](a: static int; b: Int[V]): auto {.inline.} = Int[op(a, V)]()
  template op*[V: static int](a: Int[V]; b: int): int = op(V, b)
  template op*[V: static int](a: int; b: Int[V]): int = op(a, V)

genBinOp(`max`)

# ── kernel ───────────────────────────────────────────────────────────────
# Output buffer MUST be the first kernel param (the harness prepends res).
# The dynamic value comes from an input buffer so the same kernel shape works
# on every backend.
const kernelCode = cuda:
  proc staticIntMax(res: ptr UncheckedArray[int32];
                    dyn: ptr UncheckedArray[int32]) {.global.} =
    let h = int(dyn[0])
    # func overloads (literal operand) — empty Int[N] tags forwarded to CUDA max
    let a = max(Int[5](), 7)    # Int[7]
    let b = max(7, Int[5]())    # Int[7]
    let c = max(Int[5](), 3)    # Int[5]
    let d = max(3, Int[5]())    # Int[5]
    # both-Int template — folded to Int[max(V,U)]() at Nim level
    let e = max(Int[2](), Int[3]())  # Int[3]
    let f = max(Int[3](), Int[2]())  # Int[3]
    # runtime templates — plain int max (value substituted at Nim sem)
    let g = max(Int[5](), h)    # max(5, h)
    let i = max(h, Int[5]())    # max(h, 5)
    res[0] = int32(toIntVal a)  # 7
    res[1] = int32(toIntVal b)  # 7
    res[2] = int32(toIntVal c)  # 5
    res[3] = int32(toIntVal d)  # 5
    res[4] = int32(toIntVal e)  # 3
    res[5] = int32(toIntVal f)  # 3
    res[6] = int32(g)           # 100
    res[7] = int32(i)           # 100

suite "NVRTC — static-int ambiguous-builtin (max) overload set":

  test "all five overload shapes compile, run, and produce the right values":
    var buf: array[8, int32]
    var dynArr: array[1, int32] = [100'i32]
    var nv = initNvrtc(kernelCode)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()  # currently aborts: NVRTC rejects the empty-Int[N] max calls (see header)
    nv.getPtx()
    nv.execute("staticIntMax", buf, (dynArr,))
    check buf[0] == 7    # max(Int[5](), 7)
    check buf[1] == 7    # max(7, Int[5]())
    check buf[2] == 5    # max(Int[5](), 3)
    check buf[3] == 5    # max(3, Int[5]())
    check buf[4] == 3    # max(Int[2](), Int[3]())
    check buf[5] == 3    # max(Int[3](), Int[2]())
    check buf[6] == 100  # max(Int[5](), 100)
    check buf[7] == 100  # max(100, Int[5]())
