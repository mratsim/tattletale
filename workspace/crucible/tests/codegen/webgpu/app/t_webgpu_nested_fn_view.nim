## WGSL nested device fns: pointer-field bindings resolve per call site.
##
## Test that `inner(y, x)` called as `inner(x, y)` writes through the buffer
## passed as first argument, with and without param/caller name collisions.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/app/t_webgpu_nested_fn_view.nim

import workspace/crucible

type
  AxpyView = object
    p: ptr UncheckedArray[float32]
    n: int

  AxpyBuf[DevMem] = object
    p: DevMem
    n: int

proc innerSwap(y, x: AxpyView) =
  y.p[0] = x.p[0] + 1

proc innerPlain(dst, src: AxpyView) =
  dst.p[0] = src.p[0] + 100

proc outerSwap(y: AxpyView, x: AxpyView) =
  innerSwap(x, y)

proc outerPlain(y: AxpyView, x: AxpyView) =
  innerPlain(x, y)

proc runTest() =
  var engine = bkWGSL.init()
  const N = 4

  # Swap variant: the callee's param names collide with the caller's
  # variable names and the arguments are swapped.
  block:
    let xDev = engine.allocDevice(float32, N)
    let yDev = engine.allocDevice(float32, N)
    var xHost: array[N, float32]
    var yHost: array[N, float32]
    for i in 0 ..< N:
      xHost[i] = float32(i + 1)
      yHost[i] = float32(10 * (i + 1))
    engine.copyToDeviceFromHost(xDev, addr xHost[0], N)
    engine.copyToDeviceFromHost(yDev, addr yHost[0], N)

    engine.launchDeviceFn outerSwap(AxpyBuf[typeof(yDev)](p: yDev, n: N),
                                    AxpyBuf[typeof(xDev)](p: xDev, n: N)) do:
      grid = (1, 1)
      blk = (1, 1)

    var gotX: array[N, float32]
    var gotY: array[N, float32]
    engine.copyToHostFromDevice(addr gotX[0], xDev, N)
    engine.copyToHostFromDevice(addr gotY[0], yDev, N)
    doAssert gotX[0] == yHost[0] + 1,
      "swap: inner's y param binds to the x buffer and reads y, got x[0]=" & $gotX[0] &
      " want " & $(yHost[0] + 1) & " y[0]=" & $gotY[0]
    doAssert gotY[0] == yHost[0],
      "swap: the y buffer must be untouched, got y[0]=" & $gotY[0]

  # No-collision variant: the callee's params share no name with any
  # caller variable. The per-variable lookup misses, so both fields
  # resolve through the single-slot type-keyed fallback, whose answer
  # is the last-constructed buffer. The value betrays the wrong source.
  block:
    let xDev = engine.allocDevice(float32, N)
    let yDev = engine.allocDevice(float32, N)
    var xHost: array[N, float32]
    var yHost: array[N, float32]
    for i in 0 ..< N:
      xHost[i] = float32(i + 1)
      yHost[i] = float32(10 * (i + 1))
    engine.copyToDeviceFromHost(xDev, addr xHost[0], N)
    engine.copyToDeviceFromHost(yDev, addr yHost[0], N)

    engine.launchDeviceFn outerPlain(AxpyBuf[typeof(yDev)](p: yDev, n: N),
                                     AxpyBuf[typeof(xDev)](p: xDev, n: N)) do:
      grid = (1, 1)
      blk = (1, 1)

    var gotX: array[N, float32]
    var gotY: array[N, float32]
    engine.copyToHostFromDevice(addr gotX[0], xDev, N)
    engine.copyToHostFromDevice(addr gotY[0], yDev, N)
    doAssert gotX[0] == yHost[0] + 100,
      "plain: dst=x must receive src=y's value, got x[0]=" & $gotX[0] &
      " want " & $(yHost[0] + 100) & " y[0]=" & $gotY[0]

  echo "[OK] nested device fns: pointer-field bindings resolve per call site"

runTest()
