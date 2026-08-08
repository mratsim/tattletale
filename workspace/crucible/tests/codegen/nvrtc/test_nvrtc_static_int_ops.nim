## NVRTC: `+`/`*` on Int[N] with compile-time and dynamic int operands
##
## Int[N] is a CuTe-style compile-time integer: its value lives in the type
## parameter (crucible-only — the overloads below are hand-written to mirror
## ceramic's int_tuples_datatypes.nim). Two cases:
##   - compile-time int (a static-int type or an int literal): the result is a
##     static-int type (Int[10]() + 1 -> Int[11]); everything is handled at the
##     Nim level and the C compiler can optimize the code away.
##   - dynamic int (a runtime value): the codegen materializes the type-level
##     integer into a literal int (Int[10]() + h -> 10 + h).
##
## The five cases tested, for both + and *:
##   1. both operands Int[N] types:  Int[V]() + Int[U]()  -> Int[V+U]() (resolved at Nim compile time)
##   2. literal on the right:        Int[V]() + 1         -> Int[V+1]()
##   3. literal on the left:         2 + Int[V]()         -> Int[2+V]()
##   4. runtime value on the right:  Int[V]() + h         -> V + h (plain int)
##   5. runtime value on the left:   h + Int[V]()         -> h + V (plain int)
## Static-int results (cases 1-3) are verified via toIntVal (compile-time
## extraction from the type); dynamic results (cases 4-5) are plain ints.
##
## Run:
##   cd tattletale
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##     nim cpp -r --hints:off --warnings:off \
##       --outdir:build/tests/nvrtc --nimcache:nimcache/tests/nvrtc \
##       workspace/crucible/tests/codegen/nvrtc/test_nvrtc_static_int_ops.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc

# ── static-int type + genBinOp overload set (mirrors ceramic) ────────────
type Int*[V: static int] = object

template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V

template genBinOp(op: untyped): untyped =
  template op*[V, U: static int](a: Int[V]; b: Int[U]): auto = Int[op(V, U)]()
  func op*[V: static int](a: Int[V]; b: static int): auto {.inline.} = Int[op(V, b)]()
  func op*[V: static int](a: static int; b: Int[V]): auto {.inline.} = Int[op(a, V)]()
  template op*[V: static int](a: Int[V]; b: int): int = op(V, b)
  template op*[V: static int](a: int; b: Int[V]): int = op(a, V)

genBinOp(`+`)
genBinOp(`*`)

# ── kernel ───────────────────────────────────────────────────────────────
# Output buffer MUST be the first kernel param (the harness prepends res).
# The dynamic value comes from an input buffer so the same kernel shape works
# on every backend.
const kernelCode = cuda:
  proc staticIntOps(res: ptr UncheckedArray[int32];
                    dyn: ptr UncheckedArray[int32]) {.global.} =
    let h = int(dyn[0])
    # func overloads (literal operand) — empty Int[N] tags, contained device fns
    let a = Int[10]() + 1        # Int[11]
    let b = 2 + Int[10]()        # Int[12]
    let c = Int[10]() * 3        # Int[30]
    let d = 4 * Int[10]()        # Int[40]
    # both-Int template — folded to Int[V op U]() at Nim level
    let e = Int[2]() + Int[3]()  # Int[5]
    let f = Int[2]() * Int[3]()  # Int[6]
    # runtime templates — plain int arithmetic (value substituted at Nim sem)
    let g = Int[10]() + h        # 10 + h
    let i = h + Int[10]()        # h + 10
    let j = Int[10]() * h        # 10 * h
    let k = h * Int[10]()        # h * 10
    res[0] = int32(toIntVal a)   # 11
    res[1] = int32(toIntVal b)   # 12
    res[2] = int32(toIntVal c)   # 30
    res[3] = int32(toIntVal d)   # 40
    res[4] = int32(toIntVal e)   # 5
    res[5] = int32(toIntVal f)   # 6
    res[6] = int32(g)            # 110
    res[7] = int32(i)            # 110
    res[8] = int32(j)            # 1000
    res[9] = int32(k)            # 1000

suite "NVRTC — static-int +/* overload set":

  test "all five overload shapes compile, run, and produce the right values":
    var buf: array[10, int32]
    var dynArr: array[1, int32] = [100'i32]
    var nv = initNvrtc(kernelCode)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("staticIntOps", buf, (dynArr,))
    check buf[0] == 11    # Int[10]() + 1
    check buf[1] == 12    # 2 + Int[10]()
    check buf[2] == 30    # Int[10]() * 3
    check buf[3] == 40    # 4 * Int[10]()
    check buf[4] == 5     # Int[2]() + Int[3]()
    check buf[5] == 6     # Int[2]() * Int[3]()
    check buf[6] == 110   # Int[10]() + 100
    check buf[7] == 110   # 100 + Int[10]()
    check buf[8] == 1000  # Int[10]() * 100
    check buf[9] == 1000  # 100 * Int[10]()
