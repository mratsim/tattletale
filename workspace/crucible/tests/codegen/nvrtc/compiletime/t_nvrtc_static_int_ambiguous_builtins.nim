## NVRTC: `max` over the Int[N] compile-time-integer overload set, the library
## overloads body-parse like `+`/`*`, all five operand shapes resolve.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/nvrtc/compiletime/t_nvrtc_static_int_ambiguous_builtins.nim

import std/[unittest]
import workspace/crucible

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

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "NVRTC — static-int ambiguous-builtin (max) overload set":

    test "all five overload shapes compile, run, and produce the right values":
      var buf: array[8, int32]
      var dynArr: array[1, int32] = [100'i32]
      var engine = bkCuda.newEngine()
      engine.ingest(kernelCode)
      let iDev = engine.allocDevice(int32, dynArr.len)
      engine.copyToDeviceFromHost(iDev, addr dynArr[0], dynArr.len)
      let oDev = engine.allocDevice(int32, buf.len)
      engine.launchKernel<<(1, 1)>>("staticIntMax", oDev, (iDev,))
      engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
      check buf[0] == 7    # max(Int[5](), 7)
      check buf[1] == 7    # max(7, Int[5]())
      check buf[2] == 5    # max(Int[5](), 3)
      check buf[3] == 5    # max(3, Int[5]())
      check buf[4] == 3    # max(Int[2](), Int[3]())
      check buf[5] == 3    # max(Int[3](), Int[2]())
      check buf[6] == 100  # max(Int[5](), 100)
      check buf[7] == 100  # max(100, Int[5]())

when isMainModule:
  runTest()
