## WebGPU: buffer-pointer device functions inline at their call sites (v1 port).
##
## Test that a device function taking a pointer that cannot cross a WGSL
## function signature inlines at the call site and mutates the caller's storage:
##
## - a buffer-pointer callee inlines, its buffer writes read back through the kernel buffer
## - a device-space pointer callee inlines through the address of a buffer element
## - a function-space pointer callee keeps a real call and still mutates a local
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/functions/t_webgpu_pointer_param_inline.nim

import std/unittest
import workspace/crucible

proc fill(p: ptr UncheckedArray[float32], n: int32) =
  for i in 0 ..< n:
    p[i] = float32(i) * 2.0'f32

proc setOne(p: ptr float32) =
  p[] = 1.5'f32

proc bump(x: var int32) =
  x = x + 1

const kernel = webgpu:
  proc k(outBuf: ptr UncheckedArray[float32]) {.global.} =
    fill(outBuf, 4)
    setOne(addr outBuf[3])
    var v: int32 = 41
    bump(v)
    outBuf[1] = float32(v)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernel

  suite "WebGPU - pointer-param call inlining":

    test "inlined buffer pointers write through, function-space params keep real calls":
      var engine = newEngine(runWebGPU)
      engine.ingest(kernel)
      let oDev = engine.allocDevice(float32, 4)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("k", oDev, ())
      var res: array[4, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[0] == 0.0'f32
      # function-space bump ran through a real call: 41 + 1
      check res[1] == 42.0'f32
      check res[2] == 4.0'f32
      # device-space pointer wrote through the buffer element's address
      check res[3] == 1.5'f32

when isMainModule:
  runTest()
