## Metal: user operator overloads on a struct emit mangled callee names.
##
## `*=` / `+=` compound assigns and `+` infix over a user struct call
## callees named by getOperatorMangledName (`star_eq_...` shape).
## The emitted MSL contains no raw `*=` identifier and the kernels compute:
##
## - two structs each overloading `+` get two distinct callees, a constant
##   hash would emit one definition twice
## - the compound assigns take a `var T` first parameter, registration
##   resolves `var T` params
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_operator_mangle.nim

import std/[strutils, unittest]
import workspace/crucible

type Acc = object
  sum: int32
  scale: int32

proc `*=`(a: var Acc, k: int32) {.device.} =
  a.sum = a.sum * k
  a.scale = a.scale * k

proc `+=`(a: var Acc, b: Acc) {.device.} =
  a.sum = a.sum + b.sum
  a.scale = a.scale + b.scale

proc `+`(a, b: Acc): Acc {.device.} =
  Acc(sum: a.sum + b.sum, scale: a.scale + b.scale)

type Pair = object
  x: int32
  y: int32

proc `+`(a, b: Pair): Pair {.device.} =
  Pair(x: a.x + b.x, y: a.y + b.y)

const opMsl = metal:
  proc compoundKernel(outp: ptr UncheckedArray[int32]) {.global.} =
    var acc = Acc(sum: 3, scale: 2)
    acc *= 5                # star_eq_<hash> callee
    acc += Acc(sum: 1, scale: 1)  # plus_eq_<hash> callee from the overload
    outp[0] = acc.sum       # (3*5)+1 = 16
    outp[1] = acc.scale     # (2*5)+1 = 11

  proc infixKernel(outp: ptr UncheckedArray[int32]) {.global.} =
    let a = Acc(sum: 10, scale: 4)
    let b = Acc(sum: 7, scale: 3)
    let c = a + b           # plus_<hash> callee from the infix overload
    outp[0] = c.sum         # 10 + 7 = 17
    outp[1] = c.scale       # 4 + 3 = 7

  proc overloadPairKernel(outp: ptr UncheckedArray[int32]) {.global.} =
    # two `+` overloads in one compilation unit: a constant hash would
    # emit one `plus_` definition twice, the MSL would not compile
    let p = Acc(sum: 0, scale: 0) + Acc(sum: 0, scale: 0)
    let q = Pair(x: 3, y: 4) + Pair(x: 5, y: 6)
    outp[0] = p.sum + int32(q.x) + int32(q.y)  # 0 + 8 + 10 = 18
    outp[1] = p.scale + int32(q.y) - int32(q.x)  # 0 + 10 - 8 = 2

proc runTest() =
  suite "Metal - operator overload mangling":
    test "struct compound assign calls a spelled-out callee":
      # the emitted source must not carry a raw `*=` or `+=` identifier
      doAssert not opMsl.contains("*="), "raw '*=' leaked into the MSL: " & opMsl
      doAssert not opMsl.contains("+="), "raw '+=' leaked into the MSL: " & opMsl
      doAssert opMsl.contains("star_eq_"), "expected the star_eq_ callee in: " & opMsl
      doAssert opMsl.contains("plus_eq_"), "expected the plus_eq_ callee in: " & opMsl
      var engine = newEngine(runMetal)
      engine.ingest(opMsl)
      var res: array[2, int32]
      let oDev = engine.allocDevice(int32, 2)
      engine.launchKernel("compoundKernel", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 16
      check res[1] == 11

    test "two structs overloading + get two distinct callees":
      doAssert not opMsl.contains("`+`"), "raw '`+`' leaked into the MSL: " & opMsl
      var engine = newEngine(runMetal)
      engine.ingest(opMsl)
      var res: array[2, int32]
      let oDev = engine.allocDevice(int32, 2)
      engine.launchKernel("overloadPairKernel", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 18
      check res[1] == 2

    test "struct infix operator calls a spelled-out callee":
      doAssert opMsl.contains("plus_"), "expected the plus_ callee in: " & opMsl
      var engine = newEngine(runMetal)
      engine.ingest(opMsl)
      var res: array[2, int32]
      let oDev = engine.allocDevice(int32, 2)
      engine.launchKernel("infixKernel", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 17
      check res[1] == 7

when isMainModule:
  runTest()
