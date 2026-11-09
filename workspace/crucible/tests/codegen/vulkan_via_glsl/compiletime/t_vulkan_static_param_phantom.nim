## Vulkan (GLSL): static param substitution over phantom-typed leaves.
##
## Test that a static object param's substituted value keeps phantom-typed and int-tuple leaf values
##
## - the substituted read must produce the phantom's static parameter
## - a runtime-typed field read materializes the whole substituted value
##
## Run:
##   cd tattletale (worktree root)
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/t_vulkan_static_param_phantom \
##     --nimcache:nimcache/tests/t_vulkan_static_param_phantom \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/compiletime/t_vulkan_static_param_phantom.nim

import workspace/crucible

type
  Phantom[V: static int] = object
  Layout = object
    shape: ((Phantom[4], Phantom[8]), (Phantom[4], Phantom[2]))
    stride: ((Phantom[0], Phantom[4]), (Phantom[1], Phantom[32]))
  ThreadMap = object
    layout: Layout
    tileShape: (int32, int32)

template value[P: static int](p: Phantom[P]): int32 = int32(P)

const mapConst = ThreadMap(
  layout: Layout(
    shape: ((Phantom[4](), Phantom[8]()), (Phantom[4](), Phantom[2]())),
    stride: ((Phantom[0](), Phantom[4]()), (Phantom[1](), Phantom[32]()))),
  tileShape: (32'i32, 2'i32))

proc storeMap(outBuf: ptr UncheckedArray[int32], map: static ThreadMap, t: int32) {.device.} =
  # static reads fold at sem time, the phantom values through their type param
  outBuf[0] = value(map.layout.shape[0][0])
  outBuf[1] = value(map.layout.shape[0][1])
  outBuf[2] = value(map.layout.shape[1][0])
  outBuf[3] = value(map.layout.shape[1][1])
  outBuf[4] = value(map.layout.stride[0][0])
  outBuf[5] = value(map.layout.stride[0][1])
  outBuf[6] = value(map.layout.stride[1][0])
  outBuf[7] = value(map.layout.stride[1][1])
  # a runtime-typed field read materializes the whole substituted value
  outBuf[8] = map.tileShape[0]
  outBuf[9] = map.tileShape[1]
  outBuf[10] = t

const kernelSrc = glsl:
  proc mapKernel(outBuf: ptr UncheckedArray[int32]) {.global.} =
    storeMap(outBuf, mapConst, 7'i32)

proc runTest() =
  var engine = newEngine(runVulkan)
  engine.ingest(kernelSrc)
  let outDev = engine.allocDevice(int32, 11)
  engine.launchKernel("mapKernel", outDev, ())
  var res: array[11, int32]
  engine.copyToHostFromDevice(addr res[0], outDev, 11)
  let expected: array[11, int32] = [4'i32, 8, 4, 2, 0, 4, 1, 32, 32, 2, 7]
  for i in 0 ..< 11:
    doAssert res[i] == expected[i],
      "res[" & $i & "] = " & $res[i] & ", expected " & $expected[i]
  echo "t_vulkan_static_param_phantom: OK"

when isMainModule:
  runTest()
