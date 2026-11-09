## launchDeviceFn with a view whose layout leaves are phantom-only structs.
##
## - a static (recursively fieldless) layout leaf contributes no kernel parameter
## - the real arguments bind at consecutive buffer slots
## - the runtime-int layout variant keeps working
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_static_leaf_params.nim

import std/unittest
import workspace/crucible

type
  IdPh[V: static int] = object
  Lay[Sh, St] = object
    shape*: Sh
    stride*: St
  DevView[Mem, Sh, St] = object
    data*: Mem
    layout*: Lay[Sh, St]
  Kv[T, Sh, St] = object
    data*: ptr UncheckedArray[T]
    layout*: Lay[Sh, St]

  ShStatic = tuple[x: IdPh[8]]
  StStatic = tuple[x: IdPh[1]]
  ShRun = tuple[x: int]
  StRun = tuple[x: int]

template toInt[V: static int](x: IdPh[V]): int = V
template toInt(x: int): int = x

proc `=destroy`[Mem, Sh, St](v: var DevView[Mem, Sh, St]) {.inline.} =
  ## Non-owning view: releases nothing.
  discard

template `()`[T, Sh, St](v: Kv[T, Sh, St], i: int): untyped =
  v.data[i * toInt(v.layout.stride[0])]

proc kCopyStatic(dst, src: distinct Kv) {.inline.} =
  let n = toInt(dst.layout.shape[0])
  var i = int(get_global_id(0))
  if i < n:
    dst(i) = src(i) + 1.0'f32

proc kCopyRun(dst, src: distinct Kv) {.inline.} =
  let n = toInt(dst.layout.shape[0])
  var i = int(get_global_id(0))
  if i < n:
    dst(i) = src(i) + 1.0'f32

proc makeViewStatic[H, T](mem: DeviceMem[H, T]): auto =
  let lay = Lay[ShStatic, StStatic](shape: (IdPh[8](),), stride: (IdPh[1](),))
  DevView[DeviceMem[H, T], ShStatic, StStatic](
    data: DeviceMem[H, T](mem: mem.mem),
    layout: lay)

proc makeViewRun[H, T](mem: DeviceMem[H, T], n: int): auto =
  let lay = Lay[ShRun, StRun](shape: (n,), stride: (1,))
  DevView[DeviceMem[H, T], ShRun, StRun](
    data: DeviceMem[H, T](mem: mem.mem),
    layout: lay)

proc runTest() =
  suite "Metal - static layout leaves contribute no kernel parameter":

    test "phantom-only layout leaves read through the real buffer slots":
      var engine = newEngine(runMetal)
      let srcDev = engine.allocDevice(float32, 8)
      let dstDev = engine.allocDevice(float32, 8)
      var host: array[8, float32] = [1.0'f32, 2, 3, 4, 5, 6, 7, 8]
      engine.copyToDeviceFromHost(srcDev, host.toOpenArray(0, 8 - 1))
      let srcView = makeViewStatic(srcDev)
      let dstView = makeViewStatic(dstDev)
      engine.launchDeviceFn kCopyStatic(dstView, srcView) do:
        grid = (1, 1)
        blk = (8, 1)
      var back: array[8, float32]
      engine.copyToHostFromDevice(back.toOpenArray(0, 8 - 1), dstDev)
      check back == [2.0'f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    test "runtime-int layout leaves keep working":
      var engine = newEngine(runMetal)
      let srcDev = engine.allocDevice(float32, 8)
      let dstDev = engine.allocDevice(float32, 8)
      var host: array[8, float32] = [1.0'f32, 2, 3, 4, 5, 6, 7, 8]
      engine.copyToDeviceFromHost(srcDev, host.toOpenArray(0, 8 - 1))
      let srcView = makeViewRun(srcDev, 8)
      let dstView = makeViewRun(dstDev, 8)
      engine.launchDeviceFn kCopyRun(dstView, srcView) do:
        grid = (1, 1)
        blk = (8, 1)
      var back: array[8, float32]
      engine.copyToHostFromDevice(back.toOpenArray(0, 8 - 1), dstDev)
      check back == [2.0'f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

runTest()
