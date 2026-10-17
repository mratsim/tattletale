## launchDeviceFn on Metal: a struct parameter enters the kernel as leaves.
## The body rebuilds the struct from them.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_struct_params.nim

import workspace/crucible

type CopyArgs = object
  src: ptr UncheckedArray[uint32]
  n: int32

proc structCopy(outP: ptr UncheckedArray[uint32], v: CopyArgs) =
  for i in 0 ..< v.n:
    outP[i] = v.src[i]

proc runTest() =
  var engine = bkMetal.init()

  block structParam:
    var a: array[4, uint32] = [3'u32, 1'u32, 4'u32, 1'u32]
    let aDev = engine.allocDevice(a.len * sizeof(uint32))
    let oDev = engine.allocDevice(a.len * sizeof(uint32))
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
    var zero: array[4, uint32] = [0'u32, 0'u32, 0'u32, 0'u32]
    engine.copyToDeviceFromHost(oDev, addr zero[0], a.len * sizeof(uint32))
    engine.launchDeviceFn structCopy(cast[ptr UncheckedArray[uint32]](oDev),
        CopyArgs(src: cast[ptr UncheckedArray[uint32]](aDev), n: 4)) do:
      grid = (1, 1)
      blk = (1, 1)
    var out32: array[4, uint32]
    engine.copyToHostFromDevice(addr out32[0], oDev, a.len * sizeof(uint32))
    engine.freeDevice(aDev)
    engine.freeDevice(oDev)
    doAssert out32 == [3'u32, 1'u32, 4'u32, 1'u32],
      "structCopy: got " & $out32
    echo "[OK] launchDeviceFn: struct parameter binds pointer leaf as buffer, packs scalar leaf as bytes"

runTest()
