## SPIR-V direct: push-block object head and tail padding host-device agreement
##
## Test that kernel parameter objects read back exact values on MoltenVK
## - a tail-rounded object parameter and a trailing scalar parameter
## - an object whose alignment exceeds its first leaf's, preceded by a narrower scalar
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_push_block_padding.nim

import workspace/crucible

type
  ObjView = object
    ## kernel-side view: device pointer + one u32 scalar
    p: ptr UncheckedArray[uint32]
    flag: uint32

const padSpv = spirv:
  proc padKernel(output: ptr UncheckedArray[uint32], o: ObjView, z: uint32) {.global.} =
    output[0] = o.p[0] + o.flag + z
    output[1] = z

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  type
    ObjBuf = object
      ## host-side arg: the DeviceMem handle the engine packs
      p: typeof(engine.allocDevice(uint32, 1))
      flag: uint32
  engine.ingest(padSpv)
  let bufDev = engine.allocDevice(uint32, 2)
  let outDev = engine.allocDevice(uint32, 2)
  var bufHost: array[2, uint32] = [100'u32, 200'u32]
  engine.copyToDeviceFromHost(bufDev, addr bufHost[0], bufHost.len)

  # ObjBuf packs as a DeviceMem leaf (8 bytes, align 8) plus the u32
  # flag for 12 bytes total. The host serialization rounds the object
  # end to align 8 so the object occupies 16 bytes and the next
  # parameter z packs at offset 16. The device push block renders
  # the flattened leaves flat (u64 @0, u32 flag @8) and the next member
  # takes offset 12 unless the padding pass pads the object tail
  engine.launchKernel("padKernel", outDev, (ObjBuf(p: bufDev, flag: 7'u32), 99'u32))

  var out32: array[2, uint32]
  engine.copyToHostFromDevice(addr out32[0], outDev, 2)
  doAssert out32[0] == 206, "o.p[0] + o.flag + z got " & $out32[0] & ", want 100 + 7 + 99 = 206"
  doAssert out32[1] == 99, "z got " & $out32[1] & ", want 99"
  echo "  object-then-scalar push block: [", out32[0], ", ", out32[1], "] exact on MoltenVK"

  block headPad:
    # ObjMix packs n at the object's align-8 head, so after the u32 scalar
    # s the host places n at offset 16 and big at 24. The device push block
    # renders the leaves flat and must hold the same offsets
    type
      ObjMix = object
        n: uint32
        big: uint64
    const headSpv = spirv:
      proc headKernel(output: ptr UncheckedArray[uint32], s: uint32, o: ObjMix) {.global.} =
        output[0] = uint32(o.big) + o.n + s
    engine.ingest(headSpv)
    let headOut = engine.allocDevice(uint32, 1)
    engine.launchKernel("headKernel", headOut, (7'u32, ObjMix(n: 5'u32, big: 0x1_0000_0002'u64)))
    var head32: array[1, uint32]
    engine.copyToHostFromDevice(addr head32[0], headOut, 1)
    doAssert head32[0] == 14, "head-padded object got " & $head32[0] & ", want low32(big) + n + s = 2 + 5 + 7 = 14"
    echo "  scalar-then-object push block: [", head32[0], "] exact on MoltenVK"

when isMainModule:
  runTest()
  echo "t_spirv_push_block_padding: all checks passed"
