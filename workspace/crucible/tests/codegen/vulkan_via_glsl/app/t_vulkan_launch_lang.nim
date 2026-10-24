## launchDeviceFn lang extra on Vulkan: the glsl text route, both launch forms.
##
## Test that the `lang = glsl` launch extra selects the glslangValidator
## ingestion route in both launch forms, with exact readbacks and the text
## artifact as the route identity.
##
## Notes:
## - the no-lang words default is gated by the recorded flattenObjectParams
##   blocker (C11D13 fix op), words-route launches of pointer-output kernels
##   refuse at the tip with a byte-identical error, before and after this diff
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/app/t_vulkan_launch_lang.nim

import workspace/crucible

type
  AddView = object
    p: ptr UncheckedArray[uint32]
    n: int

  AddBuf[DevMem] = object
    p: DevMem
    n: int

proc addSlots(y: AddView, z: uint32, a: AddView, b: AddView) =
  for i in 0 ..< y.n:
    y.p[i] = a.p[i] + b.p[i] + z

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  var aHost: array[2, uint32] = [10'u32, 20'u32]
  var bHost: array[2, uint32] = [1'u32, 2'u32]
  template freshBufs(): tuple[a, b, o: typeof(engine.allocDevice(uint32, 2))] =
    ## Fresh devices per leg, a DeviceMem handle cannot be copied across legs
    let aDev = engine.allocDevice(uint32, 2)
    let bDev = engine.allocDevice(uint32, 2)
    let oDev = engine.allocDevice(uint32, 2)
    engine.copyToDeviceFromHost(aDev, addr aHost[0], 2)
    engine.copyToDeviceFromHost(bDev, addr bHost[0], 2)
    (aDev, bDev, oDev)
  template readback(oDev: auto, msg: string) =
    var got: array[2, uint32]
    engine.copyToHostFromDevice(addr got[0], oDev, 2)
    doAssert got == [16'u32, 27'u32], msg & " got " & $got

  block:
    # do-form, lang = glsl: the printed GLSL ingests through glslangValidator
    let (aDev, bDev, oDev) = freshBufs()
    engine.launchDeviceFn addSlots(
        AddBuf[typeof(oDev)](p: oDev, n: 2), 5'u32,
        AddBuf[typeof(aDev)](p: aDev, n: 2),
        AddBuf[typeof(bDev)](p: bDev, n: 2)) do:
      grid = (1, 1)
      blk = (1, 1)
      lang = glsl
    readback(oDev, "glsl do-form")
    echo "  [OK] do-form lang = glsl: [16, 27] exact readback"

  block:
    # chevron, lang = glsl: same selection through the schedule tuple
    let (aDev, bDev, oDev) = freshBufs()
    engine.launchDeviceFn<<(1, 1, lang = glsl)>> addSlots(
      AddBuf[typeof(oDev)](p: oDev, n: 2), 5'u32,
      AddBuf[typeof(aDev)](p: aDev, n: 2),
      AddBuf[typeof(bDev)](p: bDev, n: 2))
    readback(oDev, "glsl chevron")
    echo "  [OK] chevron lang = glsl: [16, 27] exact readback"

when isMainModule:
  runTest()
  echo "t_vulkan_launch_lang: all checks passed"
