## launchDeviceFn on Metal: compound assignment through a `var` scalar parameter.
##
## A `var T` scalar parameter materializes as a pointer.
## The compound-assignment target derefs it, the bare ident would be pointer arithmetic.
##
## The chain kernel, device fn, device fn binds a thread local.
## The output buffer carries the accumulated value out.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/runtime/t_launch_device_fn_var_param_compound.nim

import workspace/crucible

proc addInto(dst: var float32, x: float32) =
  dst += x

proc accumulateKernel(outP: ptr UncheckedArray[float32], x: float32) =
  var acc: float32 = 1.0'f32
  addInto(acc, x)
  outP[0] = acc

proc runTest() =
  var engine = newEngine(runMetal)
  let dev = engine.allocDevice(float32, 1)
  engine.launchDeviceFn accumulateKernel(dev, 41.0'f32) do:
    grid = (1, 1)
    blk = (1, 1)
  var host: float32
  engine.copyToHostFromDevice(addr host, dev, 1)
  doAssert host == 42.0'f32, "accumulateKernel: got " & $host & " want 42.0"
  echo "[OK] launchDeviceFn: compound assign through a var scalar parameter"

runTest()
