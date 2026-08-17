## Metal: `+`/`*` on Int[N] with compile-time and dynamic int operands.
## Int[N] is a CuTe-style compile-time integer whose value lives in the type parameter.
## The overload set below mirrors ceramic's int_tuples_datatypes:
##   - both operands Int[N] types: Int[V]() + Int[U]() -> Int[V+U]() (Nim compile time)
##   - literal on the right:       Int[V]() + 1 -> Int[V+1]()
##   - literal on the left:        2 + Int[V]() -> Int[2+V]()
##   - runtime value on the right: Int[V]() + h -> V + h (plain int)
##   - runtime value on the left:  h + Int[V]() -> h + V (plain int)
## Static-int results (cases 1-3) are verified via toIntVal
## (compile-time extraction from the type). Dynamic results (cases 4-5)
## are plain ints computed from the input buffer on the device.
## All ten outputs are asserted byte-exact after `engine.run()`.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_static_int_ops.nim

import std/unittest
import workspace/crucible

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
# The engine binds kernel args in order. Output first (binding 0), then inputs.
# The dynamic value comes from an input buffer (same shape as CUDA's).
const kernelCode = metal:
  proc staticIntOps(res: ptr UncheckedArray[int32];
                    dyn: ptr UncheckedArray[int32]) {.global.} =
    let h = int(dyn[0])
    # func overloads (literal operand): empty Int[N] tags, contained device fns
    let a = Int[10]() + 1        # Int[11]
    let b = 2 + Int[10]()        # Int[12]
    let c = Int[10]() * 3        # Int[30]
    let d = 4 * Int[10]()        # Int[40]
    # both-Int template: folded to Int[V op U]() at Nim level
    let e = Int[2]() + Int[3]()  # Int[5]
    let f = Int[2]() * Int[3]()  # Int[6]
    # runtime templates: plain int arithmetic (value substituted at Nim sem)
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

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernelCode

  suite "Metal - static-int +/* overload set":
    test "all five overload shapes compile, run, and produce the right values":
      var dynArr: array[1, int32] = [100'i32]
      var engine = bkMetal.init()
      engine.ingest(kernelCode)
      var out32: array[10, int32]
      engine.run("staticIntOps", out32, (dynArr))
      check out32[0] == 11    # Int[10]() + 1
      check out32[1] == 12    # 2 + Int[10]()
      check out32[2] == 30    # Int[10]() * 3
      check out32[3] == 40    # 4 * Int[10]()
      check out32[4] == 5     # Int[2]() + Int[3]()
      check out32[5] == 6     # Int[2]() * Int[3]()
      check out32[6] == 110   # Int[10]() + 100
      check out32[7] == 110   # 100 + Int[10]()
      check out32[8] == 1000  # Int[10]() * 100
      check out32[9] == 1000  # 100 * Int[10]()

when isMainModule:
  runTest()
